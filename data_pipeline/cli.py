"""命令行入口：数据源目录 → pairs.csv → 通用流水线。

阅读代码可从 main() 的命令分支开始：prepare 只建表，run 才会处理动作和视频。
"""
import argparse
import json
from pathlib import Path


def main():
    """解析参数并分发任务；具体的数据处理逻辑不放在命令行入口里。"""
    from data_pipeline.adapters import BUILDERS
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    # 第一类入口：只扫描数据源、建立配对关系，不解码动作、不运行视觉模型。
    prepare = commands.add_parser('prepare', help='Build a standard video/motion CSV')
    prepare.add_argument('--source', choices=sorted(BUILDERS), required=True)
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, help='Default: <root>/pairs.csv')
    prepare.add_argument('--cameras', nargs='+', default=['c01'])
    # 第二类入口：读取统一 CSV。换数据源时，主流程无需跟着目录结构改写。
    run = commands.add_parser('run', help='Process pairs from CSV')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--csv', type=Path, help='Override input_csv in config')
    run.add_argument('--until', choices=['inventory', 'process', 'export'], default='export')
    run.add_argument('--video-device', help='Person detector device: cpu or GPU index')
    run.add_argument('--limit', type=int, help='Debug first N sequences in a separate subset_N folder')
    run.add_argument('--workers', type=int, default=1, help='Concurrent sequences; default 1 for debugging')
    run.add_argument('--force', action='store_true', help='Replace generated outputs; originals stay untouched')
    # 独立裁剪入口方便调试纯 RGB 视频，不要求已有动作标签。
    crop = commands.add_parser('crop-video')
    crop.add_argument('--video', type=Path, required=True)
    crop.add_argument('--output-root', type=Path, required=True)
    crop.add_argument('--config', type=Path, default=Path('data_pipeline/configs/person_video.yaml'))
    crop.add_argument('--device')
    crop.add_argument('--track-id', type=int)
    crop.add_argument('--force', action='store_true')
    validate = commands.add_parser('validate')
    validate.add_argument('--dataset', type=Path, required=True)
    package = commands.add_parser('package', help='Bundle release and videos into a standalone directory')
    package.add_argument('--release', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True, help='Must not already exist')
    video_check = commands.add_parser('validate-video')
    video_check.add_argument('--directory', type=Path, required=True)
    video_check.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        # BUILDERS 是“数据源名 → CSV 构建函数”的注册表，当前注册了 aistpp。
        output = args.output or args.root / 'pairs.csv'
        pairs = BUILDERS[args.source](args.root, output, args.cameras)
        print(f'{output}: {len(pairs)} video/motion pairs')
    elif args.command == 'run':
        # 到这里才加载调度器，避免单纯建 CSV 时加载不必要的处理依赖。
        from data_pipeline.core.runner import run
        print(run(args.config, args.until, video_device=args.video_device,
                  csv_path=args.csv, limit=args.limit, force=args.force, workers=args.workers))
    elif args.command == 'crop-video':
        import yaml
        from data_pipeline.components.person_video import PersonVideoConfig, preprocess_person_video
        settings = yaml.safe_load(args.config.read_text())
        if args.device is not None:
            settings['device'] = args.device
        if args.track_id is not None:
            settings['track_id'] = args.track_id
        config = PersonVideoConfig.from_mapping(settings, Path(__file__).resolve().parents[1])
        # CLI 按视频文件名建目录；组件本身接收的则是这个确切的输出目录。
        result = preprocess_person_video(args.video, args.output_root / args.video.stem, config, force=args.force)
        print(json.dumps(result, indent=2))
    elif args.command == 'package':
        from data_pipeline.exporters.package import package_release
        print(json.dumps(package_release(args.release, args.output), indent=2))
    elif args.command == 'validate':
        from data_pipeline.exporters.omg import validate_release
        print(json.dumps(validate_release(args.dataset), indent=2))
    else:
        from data_pipeline.components.person_video import validate_person_video
        print(json.dumps(validate_person_video(args.directory, args.decode), indent=2))


if __name__ == '__main__':
    main()
