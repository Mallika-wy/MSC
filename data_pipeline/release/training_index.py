"""Build compact training windows once, alongside release publication."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import os
import tempfile
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_pipeline.io import resolve_uri, sha256_file
from data_pipeline.schema.records import FeatureRecord, ModalityLink

VERSION = 1
MOTION_COLUMNS = ['motion_id', 'sequence_id', 'actor_id', 'label_type', 'storage_uri', 'row_start', 'num_frames']
STRING_COLUMNS = MOTION_COLUMNS[:5] + ['task', 'text', 'link_id', 'feature_id']
SCHEMA = pa.schema([(k, pa.dictionary(pa.int32(), pa.string())) for k in STRING_COLUMNS] +
                   [(k, pa.int64()) for k in ('row_start', 'num_frames', 'start', 'end')])


def metadata_hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in
            [root / 'dataset.json', *sorted((root / 'meta').iterdir())] if p.is_file()}


def validate_training_index(root):
    """Publication-time freshness and payload checks; training never rehashes files."""
    root = Path(root)
    path = root / 'training/summary.json'
    summary = json.loads(path.read_text()) if path.exists() else None
    if summary is None or summary['version'] != VERSION or summary['metadata_sha256'] != metadata_hashes(root):
        return build_training_index(root)
    files = {entry['file']: entry['sha256'] for entries in summary['splits'].values() for entry in entries.values()}
    for filename, digest in files.items():
        if sha256_file(root / 'training' / filename) != digest:
            raise ValueError(f'Training index checksum mismatch: {filename}; rebuild with msc-data index')
    return summary


@dataclass(frozen=True)
class TrainingMotion:
    motion_id: str
    sequence_id: str
    actor_id: str
    label_type: str
    storage_uri: str
    row_start: int
    num_frames: int


def _text_runs(motion, annotations):
    """Annotation interval endpoints determine runs; never enumerate frames."""
    candidates = []
    for a in annotations:
        lo = max(5, int(np.ceil(a['start_sec'] * 30 - 3e-7)))
        hi = min(motion['num_frames'] - 29, int(np.floor(a['end_sec'] * 30 - 30 + 3e-7)) + 1)
        if lo < hi and a['text'].strip():
            candidates.append((lo, hi, a))
    boundaries = sorted({k for lo, hi, _ in candidates for k in (lo, hi)})
    runs = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        valid = [a for start, end, a in candidates if start <= lo and end >= hi]
        if valid:
            a = min(valid, key=lambda a: (a['end_sec'] - a['start_sec'], a['annotation_id']))
            condition = (a['text'], '', '')
            if runs and runs[-1][1] == lo and runs[-1][2] == condition:
                runs[-1] = (runs[-1][0], hi, condition)
            else:
                runs.append((lo, hi, condition))
    return runs


def _media_runs(motion, candidates, context):
    """Select links vectorially for point tokens; interval tokens use the shared query."""
    from msc.data.temporal import covers, select_tokens
    ks = np.arange(5, motion['num_frames'] - 29)
    chosen = np.full(len(ks), -1, dtype=np.int32)
    for ci, (link, feature, times) in enumerate(candidates):
        starts, ends = ks / 30, (ks + 30) / 30
        valid = (chosen < 0) & (link.motion_valid_start_sec <= starts + 1e-8) & (link.motion_valid_end_sec >= ends - 1e-8)
        positions = np.flatnonzero(valid)
        if feature.token_time_semantics == 'point' and len(times) > 1:
            # MERT has regular timestamps with occasional block-boundary gaps.
            # Prefix counts test the same support/gap conditions in bulk whenever
            # a query has a majority of ordinary spacings (hence this median).
            gaps = np.diff(times)
            spacing = np.median(gaps)
            ordinary = np.r_[0, np.cumsum(np.isclose(gaps, spacing, rtol=0, atol=1e-10))]
            large = np.r_[0, np.cumsum(gaps > 2 * spacing + link.time_scale * 1e-6)]
            lo = np.maximum(starts[positions] - context, link.motion_valid_start_sec)
            hi = np.minimum(ends[positions] + context, link.motion_valid_end_sec)
            left = np.searchsorted(times, lo * link.time_scale + link.time_offset_sec)
            right = np.searchsorted(times, hi * link.time_scale + link.time_offset_sec)
            li, ri = np.minimum(left, len(times)-1), np.maximum(right-1, 0)
            regular = (right-left > 1) & (ordinary[ri]-ordinary[li] > (right-left-1)/2)
            first = (times[li] - link.time_offset_sec) / link.time_scale
            last = (times[ri] - link.time_offset_sec) / link.time_scale
            ok = regular & (large[ri] == large[li]) & (first - spacing / link.time_scale / 2 <= starts[positions] + 1e-6) & (last + spacing / link.time_scale / 2 >= ends[positions] - 1e-6)
            chosen[positions[ok]] = ci
            positions = positions[~regular]
        for pos in positions:
            _, local = select_tokens(times, link, starts[pos], ends[pos], context, feature.token_time_semantics)
            if covers(local, 0., 1., feature.token_time_semantics,
                      feature.token_rate_hz * link.time_scale if feature.token_rate_hz else None):
                chosen[pos] = ci
    boundaries = np.r_[0, np.flatnonzero(chosen[1:] != chosen[:-1]) + 1, len(ks)]
    return [(int(ks[lo]), int(ks[hi-1])+1, ('', candidates[chosen[lo]][0].link_id,
             candidates[chosen[lo]][1].feature_id)) for lo, hi in zip(boundaries, boundaries[1:])
            if hi > lo and chosen[lo] >= 0]


def build_training_index(root, *, label_policy='train_default', audio_context_sec=.25, video_context_sec=.25):
    """Read selected metadata once; write a few shared Arrow files, never motion payloads."""
    root = Path(root)
    manifest = json.loads((root / 'dataset.json').read_text())
    provenance = json.loads((root / 'meta/provenance.json').read_text())
    label = provenance.get('train_default') if label_policy == 'train_default' else label_policy
    if not label:
        return None  # Reference-only exports do not yet declare training labels.
    motions = pq.read_table(root / 'meta/motions.parquet', columns=MOTION_COLUMNS,
                            filters=[('label_type', '=', label)])
    sequences = pq.read_table(root / 'meta/sequences.parquet', columns=['sequence_id', 'split'])
    splits = dict(zip(sequences['sequence_id'].to_pylist(), sequences['split'].to_pylist()))
    annotations = defaultdict(list)
    selected_ids = motions['motion_id']
    table = pq.read_table(root / 'meta/annotations.parquet', columns=['motion_id','annotation_id','text','start_sec','end_sec'])
    table = table.filter(pc.is_in(table['motion_id'], value_set=selected_ids))
    for a in table.to_pylist():
        annotations[a['motion_id']].append(a)
    links = defaultdict(list)
    link_records = [ModalityLink(**x) for x in pq.read_table(root / 'meta/links.parquet').to_pylist()]
    features = [FeatureRecord(**x) for x in pq.read_table(root / 'meta/features.parquet').to_pylist()]
    by_asset = defaultdict(list)
    for f in features:
        by_asset[f.asset_id].append(f)
    tokens = {}
    for link in sorted(link_records, key=lambda x: x.link_id):
        if link.alignment_status == 'unknown':
            continue
        fs = by_asset[link.asset_id]
        if len(fs) > 1:
            raise ValueError(f'Multiple feature encoders for {link.asset_id}')
        if fs:
            f = fs[0]
            if f.feature_id not in tokens:
                if link.modality == 'audio' and (f.shape[1:] != [768] or 'mert' not in f.encoder_name.lower()):
                    continue
                path = resolve_uri(root, f.timestamp_uri, manifest['external_roots'])
                if path.suffix == '.parquet':
                    table = pq.read_table(path, columns=['timestamps','timestamp_shape'], filters=[('feature_id','=',f.feature_id)])
                    times = table['timestamps'][0].values.to_numpy().reshape(table['timestamp_shape'][0].as_py())
                else:
                    times = np.load(path, allow_pickle=False)
                tokens[f.feature_id] = times
            links[link.motion_id].append((link, f, tokens[f.feature_id]))
    temp = Path(tempfile.mkdtemp(prefix='.training-', dir=root))
    contexts = dict(audio=audio_context_sec, video=video_context_sec)
    summary = dict(version=VERSION, label=label, contexts=contexts, external_roots=manifest['external_roots'],
                   links=[asdict(x) for x in link_records], features=[asdict(x) for x in features], splits={})
    batches, buffers = {}, {}
    policies = ['allow_estimated', 'verified_only'] if any(x.alignment_status == 'estimated' for x in link_records) else ['allow_estimated']
    def flush(key):
        rows = buffers[key]
        if rows:
            batches[key].append(pa.Table.from_pylist(rows, schema=SCHEMA))
            rows.clear()
    try:
        for split in ('train','val','test'):
            summary['splits'][split] = {}
            for policy in policies:
                key = (split,policy)
                filename = f'{split}-{policy}.arrow'
                batches[key] = []
                buffers[key] = []
                summary['splits'][split][policy] = dict(file=filename, spans=0, windows={})
        # Stable ordering matches the source grouping used by the previous loader.
        motions = motions.sort_by([('sequence_id','ascending'),('actor_id','ascending'),('motion_id','ascending')])
        for batch in motions.to_batches(max_chunksize=8192):
            for motion in batch.to_pylist():
                if motion['num_frames'] < 35:
                    continue
                split = splits[motion['sequence_id']]
                for policy in policies:
                    runs = {'motion_only': [(5, motion['num_frames']-29, ('','',''))],
                            'text': _text_runs(motion, annotations.get(motion['motion_id'], []))}
                    candidates = links.get(motion['motion_id'], [])
                    if policy == 'verified_only':
                        candidates = [x for x in candidates if x[0].alignment_status == 'verified']
                    for modality in ('audio','video'):
                        cs = [x for x in candidates if x[0].modality == modality]
                        if cs:
                            runs[modality] = _media_runs(motion, cs, contexts[modality])
                    key = (split,policy)
                    info = summary['splits'][split][policy]
                    for task, spans in runs.items():
                        for start, end, (text, link_id, feature_id) in spans:
                            buffers[key].append(dict(motion, task=task, start=start, end=end,
                                                     text=text, link_id=link_id, feature_id=feature_id))
                            info['spans'] += 1
                            info['windows'][task] = info['windows'].get(task,0) + end-start
                    if len(buffers[key]) >= 8192:
                        flush(key)
        for key in batches:
            flush(key)
            split, policy = key
            entry = summary['splits'][split][policy]
            table = (pa.concat_tables(batches[key]).unify_dictionaries() if batches[key]
                     else pa.Table.from_pylist([], schema=SCHEMA))
            with pa.ipc.new_file(temp / entry['file'], SCHEMA) as writer:
                writer.write_table(table)
            batches[key].clear()
            entry['sha256'] = sha256_file(temp / entry['file'])
        if len(policies) == 1:
            for entries in summary['splits'].values():
                entries['verified_only'] = entries['allow_estimated']
        summary['metadata_sha256'] = metadata_hashes(root)
        summary['identity'] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
        (temp / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        destination = root / 'training'
        if destination.exists():
            previous = root / '.training-previous'
            destination.rename(previous)
            try:
                os.rename(temp, destination)
            except BaseException:
                previous.rename(destination)
                raise
            shutil.rmtree(previous)
        else:
            os.rename(temp, destination)
        return summary
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
