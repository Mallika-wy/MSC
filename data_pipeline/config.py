"""Read dataset-centric YAML; resolve only enabled dependencies."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from data_pipeline.sources import DATASETS


@dataclass
class Settings:
    dataset: str
    root: Path
    output: Path
    retarget: dict
    tracker: dict
    audio: dict
    quality: dict
    text: dict = field(default_factory=dict)
    video: dict = field(default_factory=dict)
    release: dict = field(default_factory=dict)

    asset_manifest: Path | None = None

    @classmethod
    def read(cls, config, output=None, *, root=None):
        config = Path(config).resolve()
        values = yaml.safe_load(config.read_text())
        def path(value):
            return str((config.parent / Path(value).expanduser()).resolve())
        if 'source' not in values or any(key in values for key in ('dataset', 'root', 'retarget', 'tracker', 'output')):
            raise ValueError('Use source/motion/release configuration; legacy top-level fields are unsupported')
        source = values['source']
        dataset = source['dataset']
        if dataset not in DATASETS:
            raise ValueError(f'Unsupported dataset: {dataset}')
        motion = values.get('motion', {})
        if motion.get('target_fps', 30) != 30 or motion.get('representation', 'g1_qpos36') != 'g1_qpos36':
            raise ValueError('MSC release requires g1_qpos36 at 30 FPS')
        retarget = dict(motion.get('retarget', {}))
        tracker = dict(motion.get('tracking', {}))
        retarget.setdefault('enabled', 'auto')
        tracker.setdefault('enabled', dataset != 'omg')
        audio = dict(values.get('audio', {}))
        audio.setdefault('enabled', dataset != 'omg')
        text = dict(values.get('text', {})); text.setdefault('enabled', dataset == 'omg')
        video = dict(values.get('video', {})); video.setdefault('enabled', False)
        if retarget['enabled'] not in (True, False, 'auto'):
            raise ValueError('retarget.enabled must be true, false or auto')
        for section in (tracker, audio, text, video):
            if type(section['enabled']) is not bool:
                raise ValueError('enabled must be a boolean')
        for section, keys, enabled in (
            (retarget, ('gmr_root', 'smpl_model'), dataset != 'omg' and retarget['enabled'] is not False),
            (tracker, ('sonic_root', 'encoder', 'decoder', 'observation_config'), tracker['enabled']),
            (audio, ('model',), audio['enabled']),
            (video, ('weights',), video['enabled'])):
            if enabled:
                for key in keys:
                    if key not in section:
                        raise ValueError(f'Missing enabled stage path: {key}')
                    section[key] = path(section[key])
        release = dict(values.get('release', {}))
        release.setdefault('max_shard_frames', 1_000_000)
        release.setdefault('alignment_policy', 'allow_estimated')
        release.setdefault('split_policy', 'preserve')
        parent = 'source_g1' if dataset == 'omg' else 'retarget_reference'
        release.setdefault('train_default', 'tracker_executed' if tracker['enabled'] else parent)
        release.setdefault('publish_labels', [parent, 'tracker_executed'] if tracker['enabled'] else [parent])
        labels = release['publish_labels']
        if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)) or not set(labels) <= {'source_g1','retarget_reference','tracker_executed'}:
            raise ValueError('publish_labels must be unique supported labels')
        if type(release['max_shard_frames']) is not int or release['max_shard_frames'] < 2:
            raise ValueError('max_shard_frames must be an integer >= 2')
        if release['train_default'] not in release['publish_labels']:
            raise ValueError('train_default must be a published label')
        if 'tracker_executed' in release['publish_labels'] and not tracker['enabled']:
            raise ValueError('Executed labels require tracking')
        if 'tracker_executed' in release['publish_labels'] and parent not in release['publish_labels']:
            raise ValueError('Executed labels require their published parent')
        if release['alignment_policy'] not in ('allow_estimated', 'verified_only'):
            raise ValueError('Invalid alignment_policy')
        if release['split_policy'] not in ('preserve', 'song_disjoint'):
            raise ValueError('Invalid split_policy')
        destination = Path(output).resolve() if output else Path(path(release.get('output', 'processed')))
        root = Path(root).expanduser().resolve() if root is not None else Path(path(source['root']))
        if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError('Output must be outside the source dataset')
        return cls(dataset, root, destination, retarget, tracker, audio, values.get('quality', {}), text, video, release, Path(path(source['asset_manifest'])) if source.get('asset_manifest') else None)
