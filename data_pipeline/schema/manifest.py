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
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path(root)
        self.validate(root, release=release)
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'dataset.json').exists():
            raise FileExistsError(root / 'dataset.json')
        (root / 'meta').mkdir(exist_ok=True)
        for name, cls in TABLES.items():
            rows = [asdict(r) for r in getattr(self, name)]
            if name == 'sequences':
                for r in rows:
                    r['source_provenance'] = json.dumps(r['source_provenance'], sort_keys=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema(cls)), root / 'meta' / f'{name}.parquet', compression='zstd')
        (root / 'meta/provenance.json').write_text(json.dumps(self.provenance, indent=2, ensure_ascii=False))
        from data_pipeline.robot import robot_contract
        manifest = dict(schema_version='1.0', kind='release' if release else 'inventory', target_motion_fps=30.,
                        representation=REPRESENTATION, external_roots=self.external_roots,
                        robot=robot_contract())
        (root / 'dataset.json').write_text(json.dumps(manifest, indent=2))
        return root

    @classmethod
    def read(cls, root):
        import pyarrow.parquet as pq
        root = Path(root)
        manifest = json.loads((root / 'dataset.json').read_text())
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
        return cls(**tables, external_roots=manifest['external_roots'],
                   provenance=json.loads((root / 'meta/provenance.json').read_text())).validate(root, release=manifest['kind'] == 'release')
