"""Read-only MSC releases: deterministic task/window indices and bounded I/O."""
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from data_pipeline.schema.manifest import Inventory
from data_pipeline.release.export import read_motion
from data_pipeline.release.features import read_feature
from data_pipeline.io import resolve_uri, sha256_file
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics
from .temporal import select_tokens, covers


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ByteCache:
    def __init__(self, limit):
        self.limit, self.size, self.items = limit, 0, OrderedDict()

    def get(self, key, loader):
        if key in self.items:
            value = self.items.pop(key)
            self.items[key] = value
            return value
        value = loader()
        size = sum(x.nbytes for x in value)
        while self.items and self.size + size > self.limit:
            _, old = self.items.popitem(last=False)
            self.size -= sum(x.nbytes for x in old)
        if size <= self.limit:
            self.items[key] = value
            self.size += size
        return value


def audit_splits(inventories):
    """Audit all splits/roots, including shared audio (not enforced by v1 schema)."""
    seen = {}
    def check(key, split):
        previous = seen.setdefault(key, split)
        if previous != split:
            raise ValueError(f'Split leakage: {key}: {previous}/{split}')
    for inv in inventories:
        seq = {s.sequence_id: s for s in inv.sequences}
        motions = {m.motion_id: m for m in inv.motions}
        assets = {a.asset_id: a for a in inv.assets}
        for s in inv.sequences:
            check(('sequence', s.source_dataset, s.source_sequence_id), s.split)
            check(('group', s.source_dataset, s.split_group_id), s.split)
        for m in inv.motions:
            s = seq[m.sequence_id]
            check(('motion', s.source_dataset, m.motion_id), s.split)
            if m.parent_motion_id:
                parent = motions[m.parent_motion_id]
                if parent.actor_id != m.actor_id or parent.sequence_id != m.sequence_id:
                    raise ValueError('Parent motion Actor/Sequence mismatch')
                if seq[parent.sequence_id].split != s.split:
                    raise ValueError('Parent motion split leakage')
        for link in inv.links:
            m, a = motions[link.motion_id], assets[link.asset_id]
            if link.actor_id is not None and link.actor_id != m.actor_id:
                raise ValueError('Link Actor mismatch')
            split = seq[m.sequence_id].split
            check(('asset_checksum', a.modality, a.checksum), split)
            check(('asset_source', a.modality, a.source_asset_id), split)


class ReleaseDataset(Dataset):
    def __init__(self, roots, *, split='train', label_policy='train_default',
                 alignment_policy='allow_estimated', task_weights=None, audio_context_sec=.25,
                 video_context_sec=.25, text_cache_dir=None, cache_bytes=128 * 1024**2,
                 limit_size=None, max_motions_per_task=None, max_windows_per_motion=None):
        self.roots = [Path(p).resolve() for p in roots]
        self.inventories = [Inventory.read(p) for p in self.roots]
        if any(json.loads((p / 'dataset.json').read_text())['kind'] != 'release' for p in self.roots):
            raise ValueError('Training requires a published Release')
        audit_splits(self.inventories)
        self.cache_bytes, self.cache = cache_bytes, None
        self.codec = None
        self.text_cache_dir = Path(text_cache_dir) if text_cache_dir else None
        self.text_manifest = (json.loads((self.text_cache_dir / 'manifest.json').read_text())
                              if self.text_cache_dir else None)
        self.task_weights = {k: v for k, v in (task_weights or {'text': .5, 'audio': .5}).items() if v > 0}
        self.spans, self.ends, self.task_ranges = [], [], {k: [] for k in self.task_weights}
        self.report = {'short_motions': 0, 'windows': {}, 'excluded_conditions': 0}
        feature_identity, releases = [], []
        index_cache = ByteCache(cache_bytes)
        total = 0
        indexed_motions = {}
        chosen_identities = set()
        for ri, (root, inv) in enumerate(zip(self.roots, self.inventories)):
            metadata_hashes = {str(p.relative_to(root)): sha256_file(p) for p in
                              [root / 'dataset.json', root / 'meta/provenance.json', *sorted((root / 'meta').glob('*.parquet'))]}
            # Detect payload replacement without scanning/decompressing whole motion shards.
            uris = sorted({m.storage_uri for m in inv.motions} |
                          {uri for f in inv.features for uri in (f.feature_uri,f.timestamp_uri)})
            storage = {}
            for uri in uris:
                stat = resolve_uri(root,uri,inv.external_roots).stat()
                storage[uri] = (stat.st_size,stat.st_mtime_ns)
            releases.append(dict(metadata=metadata_hashes,storage_signatures=storage))
            label = inv.provenance.get('train_default') if label_policy == 'train_default' else label_policy
            if not label:
                raise ValueError('Missing train_default in meta/provenance.json; choose an explicit label')
            seq = {s.sequence_id: s for s in inv.sequences}
            motion_map = {m.motion_id: m for m in inv.motions}
            groups = {}
            for m in inv.motions:
                ancestor = m
                visited = set()
                while ancestor.parent_motion_id:
                    if ancestor.motion_id in visited:
                        raise ValueError('Motion parent cycle')
                    visited.add(ancestor.motion_id)
                    ancestor = motion_map[ancestor.parent_motion_id]
                groups.setdefault((m.sequence_id, m.actor_id, ancestor.motion_id), []).append(m)
            features = {}
            for f in inv.features:
                features.setdefault(f.asset_id, []).append(f)
            links, annotations = {}, {}
            for link in inv.links:
                links.setdefault(link.motion_id, []).append(link)
            for annotation in inv.annotations:
                if annotation.motion_id is None:
                    raise ValueError('Sequence-only text has ambiguous Actor/time origin; publish motion-specific annotations')
                annotations.setdefault(annotation.motion_id, []).append(annotation)
            for _,group in sorted(groups.items()):
                chosen = [m for m in group if m.label_type == label]
                if len(chosen) != 1:
                    raise ValueError(f'Expected one {label} per source Motion/Actor, found {len(chosen)}')
                m = chosen[0]
                source = seq[m.sequence_id]
                key = (source.source_dataset,source.source_sequence_id,m.actor_id,group[0].parent_motion_id or group[0].motion_id)
                if key in chosen_identities:
                    raise ValueError(f'Duplicate source Motion/Actor across Release roots: {key}')
                chosen_identities.add(key)
                if seq[m.sequence_id].split != split:
                    continue
                if m.num_frames < 35:
                    self.report['short_motions'] += 1
                    continue
                for task in self.task_weights:
                    if max_motions_per_task is not None and indexed_motions.get(task,0)>=max_motions_per_task:
                        continue
                    before = total
                    candidates = sorted(annotations.get(m.motion_id, []), key=lambda a: a.annotation_id) if task == 'text' else []
                    if task != 'text':
                        for link in sorted(links.get(m.motion_id, []), key=lambda x: x.link_id):
                            if link.modality != task or link.alignment_status == 'unknown':
                                continue
                            if alignment_policy == 'verified_only' and link.alignment_status != 'verified':
                                continue
                            fs = features.get(link.asset_id, [])
                            if len(fs) > 1:
                                raise ValueError('Multiple feature encoders for asset; provide an unambiguous Release')
                            if fs:
                                f = fs[0]
                                values, times = index_cache.get((ri,f.feature_id),lambda: read_feature(root, f, inv.external_roots))
                                if list(values.shape) != f.shape or str(values.dtype) != f.dtype or len(times) != len(values):
                                    raise ValueError('Feature payload shape/dtype/timestamps differ from metadata')
                                if task == 'audio' and (values.ndim != 2 or values.shape[1] != 768 or 'mert' not in f.encoder_name.lower()):
                                    raise ValueError('Audio task requires frozen MERT [T,768]')
                                if task == 'video' and (values.ndim != 3 or f.token_time_semantics != 'interval'):
                                    raise ValueError('Video requires verified [T,K,D] interval features')
                                from data_pipeline.schema.time_map import TimeMap
                                TimeMap(link.time_scale,link.time_offset_sec).select(times,0.,1.,semantics=f.token_time_semantics)
                                feature_identity.append((task, f.encoder_name, f.encoder_checkpoint_hash, f.preprocess_hash, f.shape[1:]))
                                candidates.append((link, f, times))
                    # Compact contiguous runs with the same chosen condition, not one Python object per window.
                    run_start, previous = None, None
                    for k in range(5, m.num_frames - 29):
                        start, end = k / 30, (k + 30) / 30
                        selected = None
                        if task == 'motion_only':
                            selected = ('motion_only',)
                        elif task == 'text':
                            valid = [a for a in candidates if a.text.strip() and a.start_sec <= start + 1e-8 and a.end_sec >= end - 1e-8]
                            if valid:
                                a = min(valid, key=lambda a: (a.end_sec-a.start_sec, a.annotation_id))
                                if self.text_manifest is None or fingerprint(a.text) not in self.text_manifest['entries']:
                                    raise ValueError('Missing T5 cache for a valid text window; run msc.cache_text --release-root')
                                selected = ('text', a.annotation_id, a.text)
                        else:
                            for link, f, times in candidates:
                                if link.motion_valid_start_sec > start + 1e-8 or link.motion_valid_end_sec < end - 1e-8:
                                    continue
                                # Include boundary support tokens, not just centres inside [start,end).
                                context = audio_context_sec if task == 'audio' else video_context_sec
                                _, local = select_tokens(times, link, start, end, context, f.token_time_semantics)
                                if covers(local, 0., 1., f.token_time_semantics, f.token_rate_hz * link.time_scale if f.token_rate_hz else None):
                                    selected = (task, link.link_id, f.feature_id)
                                    break
                        if selected != previous:
                            if previous is not None:
                                total = self._add_span(total, ri, m, task, run_start, k, previous)
                            run_start, previous = k, selected
                        if selected is None:
                            self.report['excluded_conditions'] += 1
                    if previous is not None:
                        total = self._add_span(total, ri, m, task, run_start, m.num_frames-29, previous)
                    if total > before:
                        indexed_motions[task] = indexed_motions.get(task,0)+1
        self.rollout_spans = self.spans.copy()
        if max_motions_per_task is not None or max_windows_per_motion is not None:
            grouped = {}
            for span in self.spans:
                ri,m,task,start,end,cond = span
                grouped.setdefault((task,ri,m.motion_id),[]).append(span)
            self.spans,self.ends = [],[]
            self.task_ranges = {k:[] for k in self.task_weights}
            counts,total = {},0
            for (task,ri,mid),spans in sorted(grouped.items()):
                if max_motions_per_task is not None and counts.get(task,0)>=max_motions_per_task:
                    continue
                counts[task] = counts.get(task,0)+1
                if max_windows_per_motion is None:
                    for ri,m,task,start,end,cond in spans:
                        total = self._add_span(total,ri,m,task,start,end,cond)
                else:
                    cumulative = np.cumsum([s[4]-s[3] for s in spans])
                    positions = np.unique(np.linspace(0,cumulative[-1]-1,min(max_windows_per_motion,cumulative[-1]),dtype=np.int64))
                    for pos in positions:
                        si = bisect_right(cumulative,pos)
                        ri,m,task,start,end,cond = spans[si]
                        k = int(start+pos-(cumulative[si-1] if si else 0))
                        total = self._add_span(total,ri,m,task,k,k+1,cond)
        for task, ranges in self.task_ranges.items():
            self.report['windows'][task] = sum(end-start for start, end in ranges)
            if not ranges:
                raise ValueError(f'No valid {task} windows: missing features/coverage or split; task is Pending')
        if limit_size is not None:
            raise ValueError('Release limit_size is ambiguous across tasks; use explicit Release subsets')
        self.audio_metadata = {'backend': 'msc_release', 'features': sorted(set(tuple(x[:4]) for x in feature_identity if x[0] == 'audio'))}
        self.feature_metadata = {'features': sorted(set((x[0], x[1], x[2], x[3], tuple(x[4])) for x in feature_identity)),
                                 'video_task': 'offline_demonstration',
                                 'text': self.text_manifest and self.text_manifest['encoder']}
        self.identity = fingerprint(dict(window_contract='msc_release_35f_link_clipped_context_v1',
                                         releases=releases, split=split, label_policy=label_policy,
                                         alignment=alignment_policy, weights=self.task_weights,
                                         text=self.text_manifest, index=[(r, m.motion_id,t,a,b,c) for r,m,t,a,b,c in self.spans],
                                         contexts=[audio_context_sec,video_context_sec]))
        self.contexts = {'audio': audio_context_sec, 'video': video_context_sec}
        self._lookup = None

    def _add_span(self, total, ri, motion, task, start, end, condition):
        self.spans.append((ri, motion, task, start, end, condition))
        self.ends.append(total + end - start)
        self.task_ranges[task].append((total, total + end-start))
        return total + end-start

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(cache=None,codec=None,_lookup=None)
        return state

    def window_identity(self, index):
        si = bisect_right(self.ends, index)
        ri, m, task, start, _, cond = self.spans[si]
        k = start + index - (self.ends[si-1] if si else 0)
        return ri, m, task, k, cond

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ri, m, task, k, condition = self.window_identity(index)
        return self._read_window(ri,m,task,k,condition)

    def read_window(self,release_index,motion_id,task,k):
        """Explicit deterministic evaluation query, independent of sparse train subsets."""
        for ri,m,t,start,end,condition in self.rollout_spans:
            if (ri,m.motion_id,t)==(release_index,motion_id,task) and start<=k<end:
                return self._read_window(ri,m,t,k,condition)
        return None

    def _read_window(self,ri,m,task,k,condition):
        root, inv = self.roots[ri], self.inventories[ri]
        if self.cache is None:
            self.cache = ByteCache(self.cache_bytes)
            self.codec = G1MotionFeatureCodec(G1Kinematics(), num_prev_states=5, canonical_frame_idx=4, rotation_representation='rot6d')
        window = replace(m, row_start=m.row_start+k-5, num_frames=35)
        qpos, times = self.cache.get(('motion',ri,m.motion_id,k), lambda: read_motion(root, window))
        if len(qpos) != 35 or not np.allclose(times, np.arange(k-5,k+30)/30, atol=1e-7,rtol=0):
            raise ValueError('Invalid motion-local window timestamps')
        qpos = torch.from_numpy(qpos.copy())[None]
        fk = self.codec.kinematics.forward_kinematics(qpos)
        history, anchor_pos, anchor_quat = self.codec.prev_state_features_from_history(qpos[:,:5], fk['body_pos_w'][:,:5], None, 30.)
        comp = self.codec.canonicalize(qpos[:,5:], fk['body_pos_w'][:,5:], None, anchor_pos, anchor_quat, 30.)
        sample = dict(motion_history=history[0], motion_future=self.codec.assemble_features(comp)[0],
                      motion_mask=torch.ones(30,dtype=torch.bool), task=task,
                      window_id=f'{ri}:{m.motion_id}:{m.actor_id}:{m.label_type}:{k}',
                      anchor_qpos=qpos[0,4].clone())
        if task == 'text':
            entry = self.text_manifest['entries'][fingerprint(condition[2])]
            path = resolve_uri(self.text_cache_dir,entry)
            if sha256_file(path) != self.text_manifest['checksums'][entry]:
                raise ValueError('Text cache checksum mismatch')
            obj = torch.load(path,map_location='cpu',weights_only=True)
            if obj['encoder'] != self.text_manifest['encoder'] or obj['text'] != condition[2]:
                raise ValueError('Text cache identity mismatch')
            sample.update(text_tokens=obj['features'], text_mask=obj['mask'])
        elif task in ('audio','video'):
            if self._lookup is None:
                self._lookup = [({x.link_id:x for x in inventory.links},{x.feature_id:x for x in inventory.features}) for inventory in self.inventories]
            link = self._lookup[ri][0][condition[1]]
            feature = self._lookup[ri][1][condition[2]]
            values, times = self.cache.get(('feature',ri,feature.feature_id), lambda: read_feature(root,feature,inv.external_roots))
            selected, local = select_tokens(times,link,k/30,(k+30)/30,self.contexts[task],feature.token_time_semantics)
            tokens = torch.from_numpy(np.array(values[selected],dtype=np.float32))
            sample[task+'_tokens'] = tokens
            sample[task+'_mask'] = torch.ones(tokens.shape[:-1],dtype=torch.bool)
            sample[task+('_times' if task=='audio' else '_intervals')] = torch.tensor(local,dtype=torch.float32)
        return sample
