"""Run dataset processing or inspect/export its results."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='Process dataset sequences and publish shared Parquet shards')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--limit', type=int)
    run.add_argument('--workers', type=int, default=1, help='Independent sample processes (default: 1)')
    run.add_argument('--sequence', help='Sequence ID or motion ID')
    run.add_argument('--output', type=Path)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--diagnostic', action='store_true', help='Execute rejected references for review')

    publish = commands.add_parser('publish', help='Rebuild release from saved artifacts without models')
    publish.add_argument('--work', type=Path, required=True)
    publish.add_argument('--out', type=Path, required=True)

    split = commands.add_parser('split-gdance', help='Split GDance into individual actors')
    split.add_argument('--root', type=Path, required=True)
    split.add_argument('--output', type=Path, required=True)

    validate = commands.add_parser('validate')
    validate.add_argument('--dataset', type=Path, required=True)
    validate.add_argument('--deep-sample', type=int, default=0)
    package = commands.add_parser('package')
    package.add_argument('--dataset', type=Path, required=True)
    package.add_argument('--out', type=Path, required=True)
    package.add_argument('--mode', choices=['bundle', 'reference'], default='bundle')

    crop = commands.add_parser('crop-video', help='Crop a person from source RGB video')
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

    if args.command == 'run':
        from data_pipeline.pipeline import run
        result = run(args.config, limit=args.limit, sequence=args.sequence, output=args.output,
                     dry_run=args.dry_run, diagnostic=args.diagnostic, workers=args.workers)
    elif args.command == 'publish':
        from data_pipeline.pipeline import publish_work
        result = publish_work(args.work, args.out) or {'published': False}
    elif args.command == 'split-gdance':
        from data_pipeline.sources.gdance.split import split_dataset
        sequences, actors = split_dataset(args.root, args.output)
        result = dict(sequences=sequences, actors=actors, output=str(args.output))
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
        settings['weights'] = str((args.config.resolve().parent / settings['weights']).resolve())
        result = preprocess_person_video(args.video, args.output_root / args.video.stem,
                                         PersonVideoConfig(**settings), force=args.force)
    else:
        from data_pipeline.video import validate_person_video
        result = validate_person_video(args.directory, args.decode)
    print(json.dumps(result, indent=2))
    if result.get('errors', 0) or result.get('motion_errors', 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
