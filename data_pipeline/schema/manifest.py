"""Read and write typed metadata tables and the dataset manifest."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

from .records import TABLES, REPRESENTATION


def arrow_schema(record_type):
    import pyarrow as pa
    integers = {'row_start', 'num_frames', 'size_bytes'}
    floats = {'fps', 'time_origin_sec', 'duration_sec', 'sample_rate_hz', 'frame_rate',
              'motion_valid_start_sec', 'motion_valid_end_sec', 'time_scale', 'time_offset_sec',
              'start_sec', 'end_sec', 'confidence', 'token_rate_hz'}
    return pa.schema([(f.name, pa.list_(pa.int64()) if f.name == 'shape' else
                       pa.int64() if f.name in integers else pa.float64() if f.name in floats else pa.string())
                      for f in fields(record_type)])


class MetadataWriter:
    """Write the six metadata tables with one shared, bounded serialization path."""
    def __init__(self, root):
        import pyarrow.parquet as pq
        folder = Path(root) / 'meta'
        folder.mkdir(parents=True, exist_ok=True)
        self.writers = {name: pq.ParquetWriter(folder / f'{name}.parquet', arrow_schema(cls), compression='zstd')
                        for name, cls in TABLES.items()}
        self.buffers = {name: [] for name in TABLES}
        self.closed = False

    def add(self, inventory):
        for name in TABLES:
            for record in getattr(inventory, name):
                row = asdict(record)
                if name == 'sequences':
                    row['source_provenance'] = json.dumps(row['source_provenance'], sort_keys=True)
                self.buffers[name].append(row)
                if len(self.buffers[name]) >= 2048:
                    self.flush(name)

    def flush(self, name):
        import pyarrow as pa
        if self.buffers[name]:
            self.writers[name].write_table(pa.Table.from_pylist(self.buffers[name], schema=arrow_schema(TABLES[name])))
            self.buffers[name].clear()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                for name in TABLES:
                    self.flush(name)
            finally:
                for writer in self.writers.values():
                    writer.close()


@dataclass
class Inventory:
    sequences: list = field(default_factory=list)
    motions: list = field(default_factory=list)
    assets: list = field(default_factory=list)
    links: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    features: list = field(default_factory=list)
    external_roots: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def validate(self, root='.', *, release=False, check_files=False):
        from .validation import validate_inventory
        return validate_inventory(self, root, release=release, check_files=check_files)

    def write(self, root, *, release=False):
        root = Path(root)
        self.validate(root, release=release)
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'dataset.json').exists():
            raise FileExistsError(root / 'dataset.json')
        writer = MetadataWriter(root)
        try:
            writer.add(self)
        finally:
            writer.close()
        self.write_manifest(root, release=release)
        return root

    def write_manifest(self, root, *, release=False):
        root = Path(root)
        (root / 'meta').mkdir(parents=True, exist_ok=True)
        (root / 'meta/provenance.json').write_text(json.dumps(self.provenance, indent=2, ensure_ascii=False))
        from data_pipeline.robot import robot_contract
        manifest = dict(package_mode=self.provenance.get('package_mode', 'reference'), features_bundled=self.provenance.get('portable', False), quality_storage='parquet_by_motion_id_v1' if all(m.quality_ref.endswith('.parquet') for m in self.motions) else 'json_by_motion_id_v1', feature_storage=self.provenance.get('feature_storage', 'parquet_by_feature_id_v1' if any(f.feature_uri.endswith('.parquet') for f in self.features) else 'npy_v1'), motion_storage='split_shards_v1', schema_version='1.0', kind='release' if release else 'inventory', target_motion_fps=30.,
                        representation=REPRESENTATION, external_roots=self.external_roots,
                        robot=robot_contract())
        (root / 'dataset.json').write_text(json.dumps(manifest, indent=2))
        return root

    @classmethod
    def read(cls, root, *, validate=True):
        import pyarrow.parquet as pq
        root = Path(root)
        manifest = json.loads((root / 'dataset.json').read_text())
        if manifest.get('quality_storage', 'json_by_motion_id_v1') not in ('json_by_motion_id_v1', 'parquet_by_motion_id_v1'):
            raise ValueError('Unsupported quality storage contract')
        if manifest.get('feature_storage', 'npy_v1') not in ('npy_v1', 'parquet_by_feature_id_v1'):
            raise ValueError('Unsupported feature storage contract')
        if manifest.get('kind') not in ('inventory', 'release'):
            raise ValueError('Invalid manifest kind')
        if manifest['schema_version'] != '1.0' or manifest['target_motion_fps'] != 30 or manifest['representation'] != REPRESENTATION:
            raise ValueError('Unsupported manifest contract')
        from data_pipeline.robot import robot_contract
        if manifest.get('robot') != robot_contract():
            raise ValueError('Robot joint order/assets mismatch')
        tables = {}
        for name, record_type in TABLES.items():
            table = pq.read_table(root / 'meta' / f'{name}.parquet')
            if table.schema != arrow_schema(record_type):
                raise ValueError(f'Invalid Arrow schema: {name}')
            rows = table.to_pylist()
            if name == 'sequences':
                for r in rows:
                    r['source_provenance'] = json.loads(r['source_provenance'])
            tables[name] = [record_type(**r) for r in rows]
        inventory = cls(**tables, external_roots=manifest['external_roots'],
                        provenance=json.loads((root / 'meta/provenance.json').read_text()))
        return inventory.validate(root, release=manifest['kind'] == 'release') if validate else inventory
