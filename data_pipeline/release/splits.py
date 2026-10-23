"""Deterministic splits keep shared songs and source groups together."""
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from data_pipeline.storage.store import Store, fingerprint


def _audio_identities(dataset, sequence_id, checksums):
    identities = set(checksums)
    if dataset == 'aistpp':
        # Different excerpts of the same AIST++ song can have different WAV hashes.
        music = next((part for part in sequence_id.split('_') if part.startswith('m')), None)
        if music:
            identities.add('aistpp:music:'+music)
    return sorted(identities)


def assign_song_splits(rows, config):
    """Rows are (sequence id, source group, audio identities); ratios count sequences."""
    ratios = config.get('split_ratios', [0.8, 0.1, 0.1])
    if len(ratios) != 3 or any(type(r) not in (int,float) or not np.isfinite(r) or r < 0 for r in ratios) or not np.isclose(sum(ratios),1):
        raise ValueError('split_ratios must be three nonnegative values summing to one')
    rows = list(rows)
    parents = {}
    def find(value):
        parents.setdefault(value,value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    for sid,group,songs in rows:
        if not songs:
            raise ValueError('song_disjoint requires audio identity')
        union('sequence:'+sid,'group:'+group)
        for song in songs:
            union('sequence:'+sid,'song:'+song)
    components = defaultdict(list)
    for sid,_,_ in rows:
        components[find('sequence:'+sid)].append(sid)
    groups = sorted((sorted(ids) for ids in components.values()),
                    key=lambda ids: fingerprint([config.get('split_seed',0),ids]))
    target = np.asarray(ratios) * len(rows)
    counts = np.floor(target).astype(int)
    for i in np.argsort(-(target-counts), kind='stable')[:len(rows)-sum(counts)]:
        counts[i] += 1
    assignments = {}
    # Small audio datasets: subset sums achieve exact counts whenever song groups allow it.
    for split, wanted in [('val',int(counts[1])), ('test',int(counts[2]))]:
        paths = {0: None}
        for index,ids in enumerate(groups):
            weight = len(ids)
            for total in sorted(list(paths), reverse=True):
                new = total + weight
                if new <= wanted and new not in paths:
                    paths[new] = (total,index)
            if wanted in paths:
                break
        chosen, total = set(), max(paths)
        while total:
            total,index = paths[total]
            chosen.add(index)
        for index in chosen:
            assignments.update(dict.fromkeys(groups[index],split))
        groups = [ids for index,ids in enumerate(groups) if index not in chosen]
    for ids in groups:
        assignments.update(dict.fromkeys(ids,'train'))
    return assignments


def song_disjoint(results, config):
    with tempfile.TemporaryDirectory(prefix='msc-split-') as directory:
        store = Store(Path(directory) / 'results.sqlite')
        pending, rows = [], []
        for i,result in enumerate(results):
            pending.append((result['name'], i, result))
            if len(pending) >= 2048:
                store.record_many(pending)
                pending.clear()
            if result['state'] in ('accepted','partial'):
                source = result['sequence']
                rows.append((result['name'],f"{source['source_dataset']}:{source['split_group_id']}",
                             _audio_identities(source['source_dataset'],result['name'],
                                 {a['checksum'] for a in result['assets'] if a['modality']=='audio'})))
        store.record_many(pending)
        assignments = assign_song_splits(rows,config)
        for result in store.results():
            if result['state'] in ('accepted','partial'):
                source = result['sequence']
                source['provenance'] = {**source.get('provenance',{}),'original_split':source['split']}
                source['split'] = assignments[result['name']]
            yield result


def resplit_release(root, *, ratios=(.8,.1,.1), seed=0):
    """Repartition an existing release; reuse audio/text features, never rerun models."""
    import pyarrow.parquet as pq
    from data_pipeline.schema import Inventory
    from data_pipeline.release.export import ReleaseWriter, read_motion
    from data_pipeline.release.validate import validate_release
    from data_pipeline.io import write_json

    root = Path(root).resolve()
    inv = Inventory.read(root)
    songs = defaultdict(set)
    motion_sequences = {m.motion_id:m.sequence_id for m in inv.motions}
    assets = {a.asset_id:a for a in inv.assets}
    for link in inv.links:
        if link.modality == 'audio':
            songs[motion_sequences[link.motion_id]].add(assets[link.asset_id].checksum)
    config = dict(split_policy='song_disjoint',split_ratios=list(ratios),split_seed=seed)
    assignments = assign_song_splits([(s.sequence_id,f'{s.source_dataset}:{s.split_group_id}',
                                      _audio_identities(s.source_dataset,s.source_sequence_id,songs[s.sequence_id]))
                                     for s in inv.sequences], config)
    inv.sequences = [replace(s,split=assignments[s.sequence_id],
                            source_provenance={**s.source_provenance,'original_split':s.split}) for s in inv.sequences]
    inv.provenance = {**inv.provenance,'split_audit':[],
                      'release_policy':{**inv.provenance.get('release_policy',{}),**config}}
    qualities = {row['motion_id']: json.loads(row['report_json'])
                 for row in pq.read_table(root / 'reports/quality-00000.parquet').to_pylist()}
    staged = root.with_name('.release-resplit')
    previous = root.with_name('.release-before-resplit')
    writer = ReleaseWriter(staged,external_roots=inv.external_roots,provenance=inv.provenance,
                           max_shard_frames=inv.provenance['release_policy'].get('max_shard_frames',1_000_000))
    try:
        writer.add(inv,{m.motion_id:(lambda m=m:read_motion(root,m)[0]) for m in inv.motions},qualities=qualities)
        writer.close()
        # Reuse in-package feature/media bytes while writing and checking the new indices.
        for folder in ('features','media'):
            if (root / folder).exists():
                shutil.copytree(root / folder,writer.temp / folder,copy_function=os.link)
        writer.publish()
        report = validate_release(staged,deep_sample=max(len(inv.motions),len(inv.features)))
        write_json(staged / 'reports/dataset_summary.json',report)
    except BaseException:
        writer.abort()
        if staged.exists():
            shutil.rmtree(staged)
        raise
    root.rename(previous)
    try:
        staged.rename(root)
    except BaseException:
        previous.rename(root)
        raise
    shutil.rmtree(previous)
    if root.name == 'release':
        config_path = root.parent / 'config.json'
        if config_path.exists():
            settings = json.loads(config_path.read_text())
            settings['release'].update(config)
            write_json(config_path,settings)
        summary_path = root.parent / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary['release'] = report
            write_json(summary_path,summary)
    return {split:sum(s.split==split for s in inv.sequences) for split in ('train','val','test')}
