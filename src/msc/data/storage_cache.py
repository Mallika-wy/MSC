"""Read-only memory mapped derivatives of published Parquet payloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data_pipeline.io import resolve_uri


def signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def save_array(directory, key, values):
    name = hashlib.sha256(key.encode()).hexdigest() + '.npy'
    target = directory / name
    temporary = target.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(target)
    return name


class StorageCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.entries = json.loads((self.directory / 'manifest.json').read_text())['sources']
        self.verified = {}

    def entry(self, source):
        key = str(source)
        if key not in self.verified:
            item = self.entries.get(key)
            self.verified[key] = item if item and item['signature'] == signature(source) else None
        return self.verified[key]

    def motion(self, source):
        item = self.entry(source)
        return (np.load(self.directory / item['motion'], mmap_mode='r', allow_pickle=False)
                if item and 'motion' in item else None)

    def feature(self, source, feature):
        item = self.entry(source)
        pair = item.get('features', {}).get(feature.feature_id) if item else None
        if pair is None:
            return None
        values, times = (np.load(self.directory / name, mmap_mode='r', allow_pickle=False) for name in pair)
        if list(values.shape) != feature.shape or str(values.dtype) != feature.dtype:
            raise ValueError('Cached feature metadata differs from published Release')
        return values, times


def convert_source(task):
    import pyarrow.parquet as pq
    source, kind, wanted, directory = task
    source = Path(source)
    before = signature(source)
    file = pq.ParquetFile(source)
    item = dict(signature=before)
    prefix = str(source) + ':' + str(before)
    if kind == 'motion':
        # Decode a shard once. The resulting float32 payload is exactly qpos36.
        column = file.read(columns=['qpos36'], use_threads=False)['qpos36'].combine_chunks()
        values = column.values.to_numpy().reshape(-1, 36)
        item['motion'] = save_array(directory, prefix + ':motion', values)
    else:
        item['features'] = {}
        for group in range(file.num_row_groups):
            stats = file.metadata.row_group(group).column(0).statistics
            if stats and stats.has_min_max and stats.min == stats.max and stats.min not in wanted:
                continue
            table = file.read_row_group(group, use_threads=False)
            for row, identity in enumerate(table['feature_id'].to_pylist()):
                if identity not in wanted:
                    continue
                feature = wanted[identity]
                if table['shape'][row].as_py() != feature.shape or table['dtype'][row].as_py() != feature.dtype:
                    raise ValueError('Packed feature metadata mismatch')
                values = table['values'][row].values.to_numpy(zero_copy_only=False).reshape(feature.shape)
                times = table['timestamps'][row].values.to_numpy(zero_copy_only=False).reshape(table['timestamp_shape'][row].as_py())
                item['features'][identity] = [save_array(directory, prefix + ':' + identity, values),
                                             save_array(directory, prefix + ':' + identity + ':times', times)]
        if set(item['features']) != set(wanted):
            raise ValueError('Feature IDs missing from cache source')
    file.close()
    if signature(source) != before:
        raise ValueError('Release payload changed while building cache')
    return str(source), item


def build_storage_cache(config, directory, workers=8, splits=('train', 'val')):
    from msc.factory import build_dataset
    directory = Path(directory).resolve()
    if any(directory.is_relative_to(Path(root).resolve()) for root in config.data.roots):
        raise ValueError('Cache must be outside read-only Release roots')
    directory.mkdir(parents=True, exist_ok=True)
    motions, features = {}, {}
    cfg = copy.deepcopy(config)
    cfg.data.cache_dir = None
    for split in splits:
        cfg.data.split = split
        dataset = build_dataset(cfg)
        table = dataset.spans.table.unify_dictionaries()
        for row in table.select(['release_index', 'storage_uri']).group_by(['release_index','storage_uri']).aggregate([]).to_pylist():
            path = str(resolve_uri(dataset.roots[row['release_index']], row['storage_uri']))
            motions[path] = (path, 'motion', {}, directory)
        needed = set(table['feature_id'].to_pylist()) - {''}
        for ri, (_, records) in enumerate(dataset._lookup):
            for identity in needed & records.keys():
                feature = records[identity]
                source = resolve_uri(dataset.roots[ri], feature.feature_uri, dataset.external_roots[ri])
                # NPY sources already support direct mmap reads.
                if source.suffix == '.parquet':
                    features.setdefault(str(source), {})[identity] = feature
    tasks = list(motions.values()) + [(path, 'feature', wanted, directory) for path, wanted in features.items()]
    sources = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (path, item) in enumerate(pool.map(convert_source, tasks), 1):
            sources[path] = item
            print(f'[MSC cache] {number}/{len(tasks)} {path}', flush=True)
    temporary = directory / 'manifest.tmp'
    temporary.write_text(json.dumps(dict(version=1, sources=sources), indent=2))
    temporary.replace(directory / 'manifest.json')


def main():
    from msc.config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    build_storage_cache(load_config(args.config, args.overrides), args.output, args.workers)


if __name__ == '__main__':
    main()
