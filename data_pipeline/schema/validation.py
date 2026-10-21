"""Data contracts: record types, identities, references, coverage and split rules."""
from dataclasses import fields
import math
import re
import types
from typing import get_args, get_origin, get_type_hints

from data_pipeline.io import resolve_uri
from .records import TABLES, REPRESENTATION
from .time_map import TimeMap


def validate_inventory(inventory, root='.', *, release=False, check_files=False):
    if release and (not inventory.motions or {s.sequence_id for s in inventory.sequences} != {m.sequence_id for m in inventory.motions}):
        raise ValueError('Release requires motion tracks for every sequence')
    ids = {}
    id_fields = ['sequence_id', 'motion_id', 'asset_id', 'link_id', 'annotation_id', 'feature_id']
    for (name, cls), key in zip(TABLES.items(), id_fields):
        records = getattr(inventory, name)
        ids[name] = {}
        hints = get_type_hints(cls)
        record_fields = fields(cls)
        for r in records:
            if not isinstance(r, cls):
                raise ValueError(f'{name}: expected {cls.__name__}')
            for f in record_fields:
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
            identity = getattr(r, key)
            if identity in ids[name]:
                raise ValueError(f'Duplicate {key}: {identity}')
            ids[name][identity] = r
            if name == 'features' and any(type(n) is not int for n in r.shape):
                raise ValueError('Feature shape must contain integers')
    def require(table, key):
        if key not in ids[table]:
            raise ValueError(f'Missing foreign key {table}: {key}')
        return ids[table][key]
    checked_uris = set()
    def uri(value):
        if value in checked_uris:
            return
        p = resolve_uri(root, value, inventory.external_roots)
        if check_files and not p.is_file():
            raise ValueError(f'Missing file: {p}')
        checked_uris.add(value)
    sources = {}
    for s in inventory.sequences:
        if not s.source_provenance:
            raise ValueError('Sequence provenance required')
        if s.split not in ('train', 'val', 'test', 'unassigned') or (release and s.split == 'unassigned'):
            raise ValueError('Invalid/unassigned split')
        source_key = (s.source_dataset, s.source_sequence_id)
        if sources.setdefault(source_key, s.split) != s.split:
            raise ValueError('Source sequence split leakage')
    for m in inventory.motions:
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
        if m.label_type == 'retarget_reference' and m.retarget_version == 'not_applicable':
            raise ValueError('Retarget reference requires retarget version')
        if m.label_type == 'tracker_executed' and (not m.parent_motion_id or m.tracker_version == 'not_applicable'):
            raise ValueError('Executed label requires reference parent and tracker version')
        uri(m.storage_uri)
        uri(m.quality_ref)
    for a in inventory.assets:
        if a.modality not in ('text', 'audio', 'video') or a.size_bytes < 0:
            raise ValueError('Invalid asset')
        if not re.fullmatch(r'[0-9a-f]{64}', a.checksum):
            raise ValueError('Asset checksum must be SHA256 hex')
        for value in (a.duration_sec, a.sample_rate_hz, a.frame_rate):
            if value is not None and value <= 0:
                raise ValueError('Asset rates/duration must be positive')
        if a.uri is not None:
            uri(a.uri)
    asset_splits = {}
    for link in inventory.links:
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
    for a in inventory.annotations:
        s = require('sequences', a.sequence_id)
        if not 0 <= a.confidence <= 1:
            raise ValueError('Invalid text confidence')
        if not 0 <= a.start_sec < a.end_sec:
            raise ValueError('Invalid annotation interval')
        if a.motion_id:
            m = require('motions', a.motion_id)
            if m.sequence_id != s.sequence_id or a.end_sec > m.num_frames / m.fps + 1e-8:
                raise ValueError('Annotation motion/coverage mismatch')
    for f in inventory.features:
        a = require('assets', f.asset_id)
        if f.input_asset_checksum != a.checksum or f.token_time_semantics not in ('point', 'interval'):
            raise ValueError('Feature identity/time semantics mismatch')
        if len(f.shape) < 2 or any(n <= 0 for n in f.shape) or f.dtype not in ('float16', 'float32', 'bfloat16'):
            raise ValueError('Invalid feature shape/dtype')
        if f.token_rate_hz is not None and f.token_rate_hz <= 0:
            raise ValueError('Invalid token rate')
        uri(f.feature_uri)
        uri(f.timestamp_uri)
    return inventory
