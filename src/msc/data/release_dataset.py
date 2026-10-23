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
from data_pipeline.release.export import read_motion
from data_pipeline.release.features import read_feature
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics
from .temporal import select_tokens


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
class IndexedSpans:
    """Arrow rows stay columnar; only a requested motion becomes a Python object."""
    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        from data_pipeline.release.training_index import TrainingMotion, MOTION_COLUMNS
        row = self.table.slice(int(index), 1).to_pylist()[0]
        motion = TrainingMotion(**{key: row[key] for key in MOTION_COLUMNS})
        condition = ('text', '', row['text']) if row['task'] == 'text' else (row['task'], row['link_id'], row['feature_id'])
        return row['release_index'], motion, row['task'], row['start'], row['end'], condition

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class ReleaseDataset(Dataset):
    def __init__(self, roots, *, split='train', label_policy='train_default',
                 alignment_policy='allow_estimated', tasks=None, audio_context_sec=.25,
                 video_context_sec=.25, cache_bytes=128 * 1024**2,
                 max_motions_per_task=None, max_windows_per_motion=None):
        import pyarrow as pa
        import pyarrow.compute as pc
        from data_pipeline.release.training_index import VERSION
        from data_pipeline.schema.records import FeatureRecord, ModalityLink
        self._settings = dict(roots=list(roots), split=split, label_policy=label_policy,
            alignment_policy=alignment_policy, tasks=tasks, audio_context_sec=audio_context_sec,
            video_context_sec=video_context_sec, cache_bytes=cache_bytes,
            max_motions_per_task=max_motions_per_task, max_windows_per_motion=max_windows_per_motion)
        self.roots = [Path(p).resolve() for p in roots]
        self.tasks = tasks if tasks is not None else ['text','audio']
        self.contexts = dict(audio=audio_context_sec, video=video_context_sec)
        self.cache_bytes, self.cache, self.codec = cache_bytes, None, None
        self._lookup, self.external_roots = [], []
        tables, identities, features = [], [], []
        for ri, root in enumerate(self.roots):
            path = root / 'training/summary.json'
            if not path.exists():
                raise FileNotFoundError(f'Missing training index: {path}. Run msc-data index --dataset {root}')
            summary = json.loads(path.read_text())
            if summary['version'] != VERSION or summary['contexts'] != self.contexts:
                raise ValueError('Training index version/context changed; rebuild with msc-data index')
            if label_policy != 'train_default' and label_policy != summary['label']:
                raise ValueError('Training label differs from published index; rebuild with --label-policy')
            entry = summary['splits'][split][alignment_policy]
            table = pa.ipc.open_file(pa.memory_map(str(root / 'training' / entry['file']), 'r')).read_all()
            table = table.filter(pc.is_in(table['task'], value_set=pa.array(self.tasks)))
            table = table.append_column('release_index', pa.array(np.full(len(table), ri, dtype=np.int32)))
            tables.append(table)
            identities.append((summary['identity'], entry['sha256']))
            self.external_roots.append(summary['external_roots'])
            self._lookup.append(({x['link_id']: ModalityLink(**x) for x in summary['links']},
                                 {x['feature_id']: FeatureRecord(**x) for x in summary['features']}))
            modalities = {x['asset_id']: x['modality'] for x in summary['links']}
            features.extend((modalities[f['asset_id']], f['encoder_name'], f['encoder_checkpoint_hash'],
                             f['preprocess_hash'], tuple(f['shape'][1:]))
                            for f in summary['features'] if f['asset_id'] in modalities
                            and modalities[f['asset_id']] in self.tasks)
        table = pa.concat_tables(tables)
        # Subsets select rows before constructing any Python Motion objects.
        if max_motions_per_task is not None:
            masks = []
            for task in self.tasks:
                task_rows = pc.equal(table['task'], task)
                ids = pc.cast(pc.unique(table.filter(task_rows)['motion_id']).slice(0, max_motions_per_task), pa.string())
                masks.append(pc.and_(task_rows, pc.is_in(table['motion_id'], value_set=ids)))
            keep = masks[0]
            for mask in masks[1:]:
                keep = pc.or_(keep, mask)
            table = table.filter(keep)
        self.rollout_spans = IndexedSpans(table)
        if max_windows_per_motion is not None:
            # Group once per motion/task, sample across all its condition spans.
            keys = pc.binary_join_element_wise(pc.cast(table['release_index'], pa.string()),
                    pc.cast(table['motion_id'], pa.string()), pc.cast(table['task'], pa.string()), ':')
            codes = pc.dictionary_encode(keys).combine_chunks().indices.to_numpy()
            selected, starts, ends = [], [], []
            boundaries = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1])+1, len(table)]
            lo_values = table['start'].to_numpy()
            hi_values = table['end'].to_numpy()
            for lo, hi in zip(boundaries, boundaries[1:]):
                if hi == lo:
                    continue
                counts = np.cumsum(hi_values[lo:hi]-lo_values[lo:hi])
                positions = np.unique(np.linspace(0, counts[-1]-1, min(max_windows_per_motion, counts[-1]), dtype=np.int64))
                rows = np.searchsorted(counts, positions, side='right')
                frames = lo_values[lo+rows] + positions - np.r_[0, counts][rows]
                selected.extend((lo+rows).tolist())
                starts.extend(frames.tolist())
                ends.extend((frames+1).tolist())
            table = table.take(pa.array(selected, type=pa.int64()))
            table = table.set_column(table.schema.get_field_index('start'), 'start', pa.array(starts, type=pa.int64()))
            table = table.set_column(table.schema.get_field_index('end'), 'end', pa.array(ends, type=pa.int64()))
        self.spans = IndexedSpans(table)
        self.ends = np.cumsum(table['end'].to_numpy()-table['start'].to_numpy())
        beginnings = np.r_[0, self.ends[:-1]]
        self.task_ranges = {}
        for task in self.tasks:
            rows = np.flatnonzero(pc.equal(table['task'], task).to_numpy())
            if not len(rows):
                raise ValueError(f'No valid {task} windows in {split}')
            self.task_ranges[task] = list(zip(beginnings[rows].tolist(), self.ends[rows].tolist()))
        self.report = dict(windows={t: sum(b-a for a,b in ranges) for t,ranges in self.task_ranges.items()})
        features = sorted(set(features))
        self.audio_metadata = dict(backend='msc_release', features=[f[:4] for f in features if f[0]=='audio'])
        self.feature_metadata = dict(features=features, video_task='offline_demonstration')
        self.identity = fingerprint(dict(index_version=VERSION, releases=identities, split=split,
            tasks=self.tasks, alignment=alignment_policy, label=label_policy,
            max_motions_per_task=max_motions_per_task, max_windows_per_motion=max_windows_per_motion))

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self):
        # Spawn workers reopen the shared files instead of pickling the Arrow tables.
        return self._settings

    def __setstate__(self, settings):
        self.__init__(**settings)

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
        for ri,m,t,start,end,condition in self.motion_spans(release_index,motion_id,task):
            if start<=k<end:
                return self._read_window(ri,m,t,k,condition)
        return None

    def motion_spans(self, release_index, motion_id, task):
        import pyarrow.compute as pc
        table = self.rollout_spans.table
        mask = pc.and_(pc.equal(table['release_index'], release_index),
                       pc.and_(pc.equal(table['motion_id'], motion_id), pc.equal(table['task'], task)))
        return IndexedSpans(table.filter(mask))

    def _read_window(self,ri,m,task,k,condition):
        root = self.roots[ri]
        if self.cache is None:
            self.cache = ByteCache(self.cache_bytes)
            self.codec = G1MotionFeatureCodec(G1Kinematics(), num_prev_states=5, canonical_frame_idx=4, rotation_representation='rot6d')
        window = replace(m, row_start=m.row_start+k-5, num_frames=35)
        qpos, times = self.cache.get(('motion',ri,m.motion_id,k), lambda: read_motion(root, window))
        qpos = torch.from_numpy(qpos.copy())[None]
        fk = self.codec.kinematics.forward_kinematics(qpos)
        history, anchor_pos, anchor_quat = self.codec.prev_state_features_from_history(qpos[:,:5], fk['body_pos_w'][:,:5], None, 30.)
        comp = self.codec.canonicalize(qpos[:,5:], fk['body_pos_w'][:,5:], None, anchor_pos, anchor_quat, 30.)
        sample = dict(motion_history=history[0], motion_future=self.codec.assemble_features(comp)[0],
                      motion_mask=torch.ones(30,dtype=torch.bool), task=task,
                      window_id=f'{ri}:{m.motion_id}:{m.actor_id}:{m.label_type}:{k}',
                      anchor_qpos=qpos[0,4].clone())
        if task == 'text':
            sample['text'] = condition[2]
        elif task in ('audio','video'):
            link = self._lookup[ri][0][condition[1]]
            feature = self._lookup[ri][1][condition[2]]
            values, times = self.cache.get(('feature',ri,feature.feature_id), lambda: read_feature(root,feature,self.external_roots[ri]))
            selected, local = select_tokens(times,link,k/30,(k+30)/30,self.contexts[task],feature.token_time_semantics)
            tokens = torch.from_numpy(np.array(values[selected],dtype=np.float32))
            sample[task+'_tokens'] = tokens
            sample[task+'_mask'] = torch.ones(tokens.shape[:-1],dtype=torch.bool)
            sample[task+('_times' if task=='audio' else '_intervals')] = torch.tensor(local,dtype=torch.float32)
        return sample
