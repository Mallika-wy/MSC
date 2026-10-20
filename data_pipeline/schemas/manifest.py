"""Explicit Arrow schemas, relational validation and safe manifest I/O."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
import re
import types
from typing import get_args, get_origin, get_type_hints
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .records import TABLES, REPRESENTATION
from .time_map import TimeMap


def resolve_uri(root, uri, external_roots=None):
    if not isinstance(uri, str) or not uri or '\\' in uri or '%' in uri or '\x00' in uri:
        raise ValueError(f'Unsafe URI: {uri!r}')
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment or (parsed.netloc and not parsed.scheme):
        raise ValueError('URI query/fragment not supported')
    base = Path(root).resolve()
    relative = uri
    if parsed.scheme:
        if parsed.scheme != 'root' or parsed.netloc not in (external_roots or {}):
            raise ValueError(f'Unknown external root: {uri}')
        base = Path(external_roots[parsed.netloc]).expanduser().resolve()
        relative = parsed.path.removeprefix('/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or relative in ('.', ''):
        raise ValueError(f'Unsafe URI: {uri}')
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f'URI escapes root: {uri}')
    return target


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
        if release and (not self.motions or {s.sequence_id for s in self.sequences} != {m.sequence_id for m in self.motions}):
            raise ValueError('Release requires motion tracks for every sequence')
        ids = {}
        id_fields = ['sequence_id', 'motion_id', 'asset_id', 'link_id', 'annotation_id', 'feature_id']
        for (name, cls), key in zip(TABLES.items(), id_fields):
            records = getattr(self, name)
            ids[name] = {}
            for r in records:
                if not isinstance(r, cls):
                    raise ValueError(f'{name}: expected {cls.__name__}')
                identity = getattr(r, key)
                if not isinstance(identity, str) or not identity.strip() or identity in ids[name]:
                    raise ValueError(f'Duplicate or empty {key}: {identity}')
                ids[name][identity] = r
                hints = get_type_hints(cls)
                for f in fields(r):
                    value = getattr(r, f.name)
                    hint = hints[f.name]
                    options = get_args(hint) if get_origin(hint) is types.UnionType else (hint,)
                    valid = False
                    for option in options:
                        origin = get_origin(option) or option
                        if origin is float:
                            valid |= type(value) in (float, int) and math.isfinite(value)
                        elif origin is int:
                            valid |= type(value) is int
                        elif origin is type(None):
                            valid |= value is None
                        else:
                            valid |= isinstance(value, origin)
                    if not valid:
                        raise ValueError(f'Invalid type/value for {name}.{f.name}: {value!r}')
                    if isinstance(value, str) and not value.strip():
                        raise ValueError(f'Empty {name}.{f.name}')
                if name == 'features' and any(type(n) is not int for n in r.shape):
                    raise ValueError('Feature shape must contain integers')
        def require(table, key):
            if key not in ids[table]:
                raise ValueError(f'Missing foreign key {table}: {key}')
            return ids[table][key]
        def uri(value):
            p = resolve_uri(root, value, self.external_roots)
            if check_files and not p.is_file():
                raise ValueError(f'Missing file: {p}')
        sources = {}
        for s in self.sequences:
            if not all((s.source_dataset, s.source_sequence_id, s.split_group_id, s.source_provenance)):
                raise ValueError('Sequence identity/provenance required')
            if s.split not in ('train', 'val', 'test', 'unassigned') or (release and s.split == 'unassigned'):
                raise ValueError('Invalid/unassigned split')
            source_key = (s.source_dataset, s.source_sequence_id)
            if sources.setdefault(source_key, s.split) != s.split:
                raise ValueError('Source sequence split leakage')
        for m in self.motions:
            require('sequences', m.sequence_id)
            if m.parent_motion_id == m.motion_id:
                raise ValueError('Motion cannot parent itself')
            if m.parent_motion_id:
                parent = require('motions', m.parent_motion_id)
                if parent.sequence_id != m.sequence_id or parent.actor_id != m.actor_id or parent.label_type == 'tracker_executed':
                    raise ValueError('Motion parent identity mismatch')
            if m.representation != REPRESENTATION or m.fps != 30. or m.row_start < 0 or m.num_frames < 2:
                raise ValueError('Invalid motion contract')
            if m.label_type not in ('source_g1', 'retarget_reference', 'tracker_executed'):
                raise ValueError('Invalid label_type')
            if not all((m.actor_id, m.group_id, m.retarget_version, m.tracker_version, m.quality_ref)):
                raise ValueError('Motion identity and processing provenance required')
            if m.label_type == 'retarget_reference' and m.retarget_version == 'not_applicable':
                raise ValueError('Retarget reference requires retarget version')
            if m.label_type == 'tracker_executed' and (not m.parent_motion_id or m.tracker_version == 'not_applicable'):
                raise ValueError('Executed label requires reference parent and tracker version')
            uri(m.storage_uri)
            uri(m.quality_ref)
        for a in self.assets:
            if a.modality not in ('text', 'audio', 'video') or a.size_bytes < 0:
                raise ValueError('Invalid asset')
            if not re.fullmatch(r'[0-9a-f]{64}', a.checksum):
                raise ValueError('Asset checksum must be SHA256 hex')
            if not all((a.source_asset_id, a.checksum, a.license_ref)):
                raise ValueError('Asset identity/checksum/license required')
            for value in (a.duration_sec, a.sample_rate_hz, a.frame_rate):
                if value is not None and value <= 0:
                    raise ValueError('Asset rates/duration must be positive')
            uri(a.uri)
        asset_splits = {}
        for link in self.links:
            m = require('motions', link.motion_id)
            a = require('assets', link.asset_id)
            if link.modality != a.modality:
                raise ValueError('Link modality mismatch')
            if not 0 <= link.motion_valid_start_sec < link.motion_valid_end_sec <= m.num_frames / m.fps + 1e-8:
                raise ValueError('Link outside motion coverage')
            mapping = TimeMap(link.time_scale, link.time_offset_sec)
            if link.alignment_status not in ('verified', 'estimated', 'unknown') or not link.alignment_source:
                raise ValueError('Invalid alignment provenance')
            if link.alignment_status != 'unknown' and a.duration_sec is not None:
                lo, hi = mapping.to_asset([link.motion_valid_start_sec, link.motion_valid_end_sec])
                if lo < -1e-8 or hi > a.duration_sec + 1e-8:
                    raise ValueError('Link outside asset coverage')
            if a.modality == 'video':
                split = require('sequences', m.sequence_id).split
                # Shared audio is allowed across splits; retain video identity checks.
                for identity in ((a.modality, a.source_asset_id), (a.modality, a.checksum)):
                    old = asset_splits.setdefault(identity, split)
                    if old != split:
                        raise ValueError('Shared asset split leakage')
        for a in self.annotations:
            s = require('sequences', a.sequence_id)
            if not a.text.strip() or not a.language or not a.source or not 0 <= a.confidence <= 1:
                raise ValueError('Invalid text annotation')
            if not 0 <= a.start_sec < a.end_sec:
                raise ValueError('Invalid annotation interval')
            if a.motion_id:
                m = require('motions', a.motion_id)
                if m.sequence_id != s.sequence_id or a.end_sec > m.num_frames / m.fps + 1e-8:
                    raise ValueError('Annotation motion/coverage mismatch')
        for f in self.features:
            a = require('assets', f.asset_id)
            if not all((f.encoder_name, f.encoder_checkpoint_hash, f.preprocess_hash, f.extraction_version, f.aggregation_method)):
                raise ValueError('Feature fingerprint required')
            if f.input_asset_checksum != a.checksum or f.token_time_semantics not in ('point', 'interval'):
                raise ValueError('Feature identity/time semantics mismatch')
            if len(f.shape) < 2 or any(n <= 0 for n in f.shape) or f.dtype not in ('float16', 'float32', 'bfloat16'):
                raise ValueError('Invalid feature shape/dtype')
            if f.token_rate_hz is not None and f.token_rate_hz <= 0:
                raise ValueError('Invalid token rate')
            uri(f.feature_uri)
            uri(f.timestamp_uri)
        return self

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
