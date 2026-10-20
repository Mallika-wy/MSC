"""Discover sources, build native MSC releases, validate and package assets."""
import argparse
from importlib import import_module
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    discover = commands.add_parser('discover')
    discover.add_argument('--source', choices=['aistpp', 'omg'], required=True)
    discover.add_argument('--root', type=Path, required=True)
    discover.add_argument('--out', type=Path, required=True)
    discover.add_argument('--limit', type=int)
    discover.add_argument('--cameras', nargs='+', default=['c01'])
    discover.add_argument('--dry-run', action='store_true')
    build = commands.add_parser('build')
    build.add_argument('--config', type=Path, required=True)
    build.add_argument('--stages', default='motion,text')
    build.add_argument('--limit', type=int)
    build.add_argument('--dry-run', action='store_true')
    validate = commands.add_parser('validate')
    validate.add_argument('--dataset', type=Path, required=True)
    validate.add_argument('--deep-sample', type=int, default=0)
    package = commands.add_parser('package')
    package.add_argument('--dataset', type=Path, required=True)
    package.add_argument('--out', type=Path, required=True)
    package.add_argument('--mode', choices=['bundle', 'reference'], default='bundle')
    crop = commands.add_parser('crop-video')
    crop.add_argument('--video', type=Path, required=True)
    crop.add_argument('--output-root', type=Path, required=True)
    crop.add_argument('--config', type=Path, required=True)
    crop.add_argument('--device')
    crop.add_argument('--track-id', type=int)
    crop.add_argument('--force', action='store_true')
    video_check = commands.add_parser('validate-video')
    video_check.add_argument('--directory', type=Path, required=True)
    video_check.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    if args.command == 'discover':
        discover_source = import_module(f'data_pipeline.sources.{args.source}').discover
        options = {'cameras': args.cameras} if args.source == 'aistpp' else {}
        result = discover_source(args.root, args.out, limit=args.limit, dry_run=args.dry_run, **options)
    elif args.command == 'build':
        from data_pipeline.runner import build
        result = build(args.config, stages=args.stages.split(','), limit=args.limit, dry_run=args.dry_run)
    elif args.command == 'validate':
        from data_pipeline.release.validate import validate_release
        result = validate_release(args.dataset, args.deep_sample)
    elif args.command == 'package':
        from data_pipeline.release.package import package_release
        result = package_release(args.dataset, args.out, mode=args.mode)
    elif args.command == 'crop-video':
        import yaml
        from data_pipeline.video import PersonVideoConfig, preprocess_person_video
        settings = yaml.safe_load(args.config.read_text())
        if args.device is not None:
            settings['device'] = args.device
        if args.track_id is not None:
            settings['track_id'] = args.track_id
        settings['weights'] = str((args.config.resolve().parent / Path(settings['weights']).expanduser()).resolve())
        config = PersonVideoConfig(**settings)
        result = preprocess_person_video(args.video, args.output_root / args.video.stem, config, force=args.force)
    else:
        from data_pipeline.video import validate_person_video
        result = validate_person_video(args.directory, args.decode)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
