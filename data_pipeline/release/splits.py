"""Deterministic song/source-group connected components, with an auditable policy."""
from pathlib import Path
import tempfile
import numpy as np
from data_pipeline.storage.store import Store, fingerprint


def song_disjoint(results, config):
    ratios = config.get('split_ratios', [0.8, 0.1, 0.1])
    if len(ratios) != 3 or any(type(r) not in (int,float) or not np.isfinite(r) or r < 0 for r in ratios) or not np.isclose(sum(ratios),1):
        raise ValueError('split_ratios must be three nonnegative values summing to one')
    parents = {}
    def find(value):
        parents.setdefault(value,value)
        root = value
        while parents[root] != root:
            root = parents[root]
        while parents[value] != value:
            following = parents[value]; parents[value] = root; value = following
        return root
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    with tempfile.TemporaryDirectory(prefix='msc-split-') as directory:
        store = Store(Path(directory) / 'results.sqlite')
        pending = []
        for i,result in enumerate(results):
            pending.append((result['name'], i, result))
            if len(pending) >= 2048:
                store.record_many(pending)
                pending.clear()
            if result['state'] not in ('accepted','partial'):
                continue
            source = result['sequence']
            group = f"group:{source['source_dataset']}:{source['split_group_id']}"
            songs = sorted({a['checksum'] for a in result['assets'] if a['modality']=='audio'})
            if not songs:
                raise ValueError('song_disjoint requires audio identity')
            for song in songs:
                union(group,'song:'+song)
        store.record_many(pending)
        component_songs = {}
        for node in parents:
            if node.startswith('song:'):
                component_songs.setdefault(find(node),[]).append(node)
        assignments = {}
        for root,songs in component_songs.items():
            value = int(fingerprint([config.get('split_seed',0),sorted(songs)])[:16],16)/2**64
            assignments[root] = 'train' if value < ratios[0] else ('val' if value < sum(ratios[:2]) else 'test')
        for result in store.results():
            if result['state'] not in ('accepted','partial'):
                yield result
                continue
            source = result['sequence']
            group = f"group:{source['source_dataset']}:{source['split_group_id']}"
            source['provenance'] = {**source.get('provenance',{}),'original_split':source['split']}
            source['split'] = assignments[find(group)]
            yield result
