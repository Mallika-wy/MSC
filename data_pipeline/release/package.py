"""Atomic, deduplicated packaging of native releases."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

from data_pipeline.io import sha256_file
from data_pipeline.schema import Inventory
from data_pipeline.io import resolve_uri
from .validate import validate_release


def finalize_output(output):
    """Keep an independently readable training release and remove processing data."""
    output = Path(output).resolve()
    root, staged, previous = output / 'release', output / '.release-bundle', output / '.release-reference'
    package_release(root, staged, include_media=False)
    inv = Inventory.read(staged)
    report = validate_release(staged, deep_sample=max(len(inv.motions), len(inv.features)))
    (staged / 'reports/dataset_summary.json').write_text(json.dumps(report, indent=2))
    root.rename(previous)
    try:
        staged.rename(root)
        validate_release(root)
    except BaseException:
        if root.exists():
            root.rename(staged)
        previous.rename(root)
        raise
    for folder in (previous, output / 'cache', output / 'work'):
        if folder.exists():
            shutil.rmtree(folder)
    for filename in ('run.log', 'run.pid', 'sonic-run.log'):
        (output / filename).unlink(missing_ok=True)
    return report


def package_release(root, output, *, mode='bundle', include_media=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if mode not in ('bundle', 'reference'):
        raise ValueError('mode must be bundle or reference')
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(root):
        raise ValueError('Package output must be outside source release')
    validate_release(root)
    inv = Inventory.read(root, validate=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    copied = {}
    def copy(uri, category):
        source = resolve_uri(root, uri, inv.external_roots)
        if source not in copied:
            digest = sha256_file(source)
            if category == 'motion' and source.parent.name in ('train', 'val', 'test'):
                category = f'{category}/{source.parent.name}'
            if source.is_relative_to(root) and category.split('/')[0] in ('motion', 'reports'):
                relative = source.relative_to(root).as_posix()
            else:
                relative = f'{category}/{digest}{source.suffix}'
            target = temp / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            copied[source] = relative
        return copied[source]
    try:
        motions = [replace(m, storage_uri=copy(m.storage_uri, 'motion'), quality_ref=copy(m.quality_ref, 'reports')) for m in inv.motions]
        features = []
        if mode == 'bundle':
            from .features import FeatureShardWriter, read_feature
            modalities = {a.asset_id: a.modality for a in inv.assets}
            writer = FeatureShardWriter(temp / 'features')
            try:
                for f in inv.features:
                    values, times = read_feature(root, f, inv.external_roots)
                    path = writer.add(modalities[f.asset_id], f.feature_id, values, times)
                    relative = path.relative_to(temp).as_posix()
                    features.append(replace(f, feature_uri=relative, timestamp_uri=relative))
            finally:
                writer.close()
        else:
            def reference(uri):
                return uri if uri.startswith('root://') else f'root://package_source/{uri}'
            features = [replace(f, feature_uri=reference(f.feature_uri), timestamp_uri=reference(f.timestamp_uri)) for f in inv.features]
        assets = []
        roots = {} if mode == 'bundle' else dict(inv.external_roots)
        if mode == 'reference':
            if 'package_source' in roots:
                raise ValueError('Reserved external alias package_source')
            roots['package_source'] = str(root)
        for a in inv.assets:
            if a.uri is None or not include_media:
                uri = None
            else:
                uri = copy(a.uri, f'media/{a.modality}') if mode == 'bundle' else (a.uri if a.uri.startswith('root://') else f'root://package_source/{a.uri}')
            assets.append(replace(a, uri=uri))
        packaged = replace(inv, motions=motions, assets=assets, features=features, external_roots=roots,
                           provenance={**inv.provenance, 'package_mode': mode, 'portable': mode == 'bundle',
                                       'media_bundled': mode == 'bundle' and include_media})
        packaged.write(temp, release=True)
        report = validate_release(temp, deep_sample=min(32, max(len(motions), len(features))))
        (temp / 'reports').mkdir(exist_ok=True)
        (temp / 'reports/dataset_summary.json').write_text(json.dumps(report, indent=2))
        os.rename(temp, output)
        return {**report, 'mode': mode, 'portable': mode == 'bundle', 'copied_files': len(copied)}
    except BaseException:
        shutil.rmtree(temp)
        raise
