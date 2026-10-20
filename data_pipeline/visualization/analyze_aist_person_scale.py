"""利用官方 COCO-17 关键点统计 AIST c01 视频中的人物占比。

只生成统计结果和示例图片，不裁剪或转码原始视频。关键点包围框只是人物范围的
近似，不是人体分割，也不能保证包含衣服等完整轮廓；本工具不参与主流水线。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import pickle
import subprocess
import zipfile

import numpy as np

from data_pipeline.core.io import write_json


QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def distribution(values):
    """汇总样本数量、均值和分位数，避免仅用平均值掩盖极端情况。"""
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()),
            'percentiles': {str(p): float(v) for p, v in zip(QUANTILES, np.percentile(values, QUANTILES))}}


def probe(path):
    """读取视频尺寸与帧率，并将 c01 文件名映射到共享动作的 cAll 序列名。"""
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json', str(path)]))
    return path.stem.replace('_c01_', '_cAll_'), result['streams'][0]


def boxes(points, width, height, threshold):
    """从 [帧数, 17, 3] 的像素坐标与置信度中计算逐帧人物包围框。"""
    good = (points[:, :, 2] >= threshold) & np.isfinite(points).all(axis=2)
    good &= (points[:, :, 0] >= 0) & (points[:, :, 0] < width)
    good &= (points[:, :, 1] >= 0) & (points[:, :, 1] < height)
    # 至少有 12 个可靠关节、一个头部关键点和双脚踝，才用于人物占比统计。
    # 手腕允许缺失，后续单独报告其覆盖率，不能把缺失当作完整观测。
    valid = (good.sum(axis=1) >= 12) & good[:, :5].any(axis=1) & good[:, 15:17].all(axis=1)
    xy = points[:, :, :2]
    lower = np.where(good[:, :, None], xy, np.inf).min(axis=1)
    upper = np.where(good[:, :, None], xy, -np.inf).max(axis=1)
    return np.concatenate((lower, upper), axis=1), valid, good


def square_crop(bounds, width, height, margin=1.2):
    """估算带余量的固定方形裁剪；放不进原图时返回 None，不额外补边。"""
    center = (bounds[:2] + bounds[2:]) / 2
    side = float(max(bounds[2:] - bounds[:2]) * margin)
    if side > min(width, height):
        return None
    origin = np.clip(center - side / 2, 0, np.array([width, height]) - side)
    return [float(origin[0]), float(origin[1]), float(origin[0] + side), float(origin[1] + side)]


def main():
    """遍历本地 c01 视频，按不同置信度统计人物占比并生成示例图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/AIST++'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted((args.source / 'videos').glob('*_c01_*.mp4'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        videos = dict(pool.map(probe, paths))
    print(f'Probed {len(videos)} videos', flush=True)
    ignored = set((args.source / 'archives/ignore_list.txt').read_text().split())
    metrics = {threshold: [] for threshold in (.1, .3, .5)}
    sequences, failures = [], []
    with zipfile.ZipFile(args.source / 'archives/keypoints2d.zip') as archive:
        for number, (sequence, metadata) in enumerate(sorted(videos.items())):
            if sequence in ignored:
                continue
            try:
                data = pickle.loads(archive.read(f'keypoints2d/{sequence}.pkl'))
                points = data['keypoints2d'][0]  # 官方视角顺序中，第 0 项对应 c01。
                if points.ndim != 3 or points.shape[1:] != (17, 3):
                    raise ValueError(f'Unsupported keypoint layout: {points.shape}')
                width, height = metadata['width'], metadata['height']
                times = np.asarray(data['timestamps']) / 1e6  # 官方微秒时间戳转换为秒。
                for threshold in metrics:
                    b, valid, good = boxes(points, width, height, threshold)
                    selected = b[valid]
                    if not len(selected):
                        continue
                    sizes = selected[:, 2:] - selected[:, :2]
                    ratios = sizes / [width, height]
                    metrics[threshold].append(np.column_stack((ratios, ratios.prod(axis=1), sizes)))
                    if threshold != .3:
                        continue
                    indices = np.flatnonzero(valid)
                    union = np.r_[selected[:, :2].min(axis=0), selected[:, 2:].max(axis=0)]
                    robust_union = np.r_[np.percentile(selected[:, :2], 1, axis=0), np.percentile(selected[:, 2:], 99, axis=0)]
                    crop = square_crop(union, width, height)
                    robust_crop = square_crop(robust_union, width, height)
                    coverage = None if robust_crop is None else float(((selected[:, :2] >= robust_crop[:2]).all(axis=1) &
                                                                      (selected[:, 2:] <= robust_crop[2:]).all(axis=1)).mean())
                    centers = (selected[:, :2] + selected[:, 2:]) / 2
                    median_area = float(np.median(ratios.prod(axis=1)))
                    representative = int(indices[np.argmin(abs(ratios.prod(axis=1) - median_area))])
                    sequences.append({
                        'sequence': sequence, 'genre': sequence.split('_')[0], 'video': str(args.source / 'videos' / (sequence.replace('_cAll_', '_c01_') + '.mp4')),
                        'width': width, 'height': height, 'frames': len(points), 'valid_frames': int(valid.sum()),
                        'valid_frame_fraction': float(valid.mean()), 'both_wrists_valid_fraction': float(good[valid, 9:11].all(axis=1).mean()),
                        'median_width_ratio': float(np.median(ratios[:, 0])), 'median_height_ratio': float(np.median(ratios[:, 1])),
                        'median_area_ratio': median_area, 'union_xyxy': union.tolist(),
                        'robust_union_xyxy': robust_union.tolist(), 'fixed_square_20pct_margin_xyxy': crop,
                        'robust_fixed_square_20pct_margin_xyxy': robust_crop,
                        'robust_crop_keypoint_box_coverage': coverage,
                        'center_p99_minus_p1_px': (np.percentile(centers, 99, axis=0) - np.percentile(centers, 1, axis=0)).tolist(),
                        'representative_frame': representative, 'representative_time_s': float(times[representative]),
                        'representative_box': b[representative].tolist(),
                    })
            except Exception as error:
                failures.append({'sequence': sequence, 'error': str(error)})
            if number % 200 == 0:
                print(f'Analyzed {number}/{len(videos)}', flush=True)
    summary = {}
    for threshold, chunks in metrics.items():
        values = np.concatenate(chunks)
        summary[str(threshold)] = {name: distribution(values[:, index]) for index, name in enumerate(
            ('width_ratio', 'height_ratio', 'area_ratio', 'width_px', 'height_px'))}
    fit = [s for s in sequences if s['fixed_square_20pct_margin_xyxy'] is not None]
    sizes = [s['fixed_square_20pct_margin_xyxy'][2] - s['fixed_square_20pct_margin_xyxy'][0] for s in fit]
    robust_fit = [s for s in sequences if s['robust_fixed_square_20pct_margin_xyxy'] is not None]
    report = {
        'scope': 'All local c01 videos, excluding official ignore list; not all nine views',
        'method': 'Official COCO-17 keypoint bounding boxes, not silhouette area; all annotation frames',
        'frame_gate': '>=12 in-image confident joints, >=1 head joint and both ankles; thresholds 0.1/0.3/0.5',
        'video_count': len(videos), 'ignored': len(set(videos) & ignored), 'analyzed_sequences': len(sequences),
        'failed_sequences': failures, 'resolution_counts': {f'{w}x{h}': sum(v['width'] == w and v['height'] == h for v in videos.values())
                                                       for w, h in {(v['width'], v['height']) for v in videos.values()}},
        'frame_weighted': summary,
        'sequence_weighted_median_area_ratio': distribution([s['median_area_ratio'] for s in sequences]),
        'valid_frame_fractions': distribution([s['valid_frame_fraction'] for s in sequences]),
        'fixed_square_with_20pct_margin_fit_count': len(fit), 'fixed_square_side_px': distribution(sizes),
        'robust_square_with_20pct_margin_fit_count': len(robust_fit),
        'robust_square_coverage': distribution([s['robust_crop_keypoint_box_coverage'] for s in robust_fit]),
        'center_motion_x_px': distribution([s['center_p99_minus_p1_px'][0] for s in sequences]),
        'center_motion_y_px': distribution([s['center_p99_minus_p1_px'][1] for s in sequences]),
        'per_genre': {genre: {'sequences': len(group), 'sequence_median_area_ratio': float(np.median([s['median_area_ratio'] for s in group]))}
                      for genre in sorted({s['genre'] for s in sequences})
                      for group in [[s for s in sequences if s['genre'] == genre]]},
    }
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'sequences.json', sequences)
    # 只提取六张代表性静帧叠加框线，不改动原始视频。
    from PIL import Image, ImageDraw
    ordered = sorted(sequences, key=lambda s: s['median_area_ratio'])
    examples = [ordered[round((len(ordered) - 1) * p)] for p in (.05, .25, .5, .75, .95)]
    examples.append(max(sequences, key=lambda s: s['center_p99_minus_p1_px'][0]))
    sheet = Image.new('RGB', (1440, 636), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(examples):
        raw = subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-ss', str(record['representative_time_s']),
                                       '-i', record['video'], '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', '-'])
        image = Image.open(BytesIO(raw)).convert('RGB')
        overlay = ImageDraw.Draw(image)
        overlay.rectangle(record['representative_box'], outline='red', width=5)
        if record['fixed_square_20pct_margin_xyxy'] is not None:
            overlay.rectangle(record['fixed_square_20pct_margin_xyxy'], outline='lime', width=5)
        image = image.resize((480, 270))
        x, y = (index % 3) * 480, (index // 3) * 318
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + 274), record['sequence'], fill='black')
        draw.text((x + 5, y + 292), f"median keypoint-box area: {record['median_area_ratio']:.1%}; red=frame, green=sequence crop", fill='black')
    sheet.save(args.output / 'examples.png')
    print(json.dumps({key: value for key, value in report.items() if key not in ('frame_weighted', 'per_genre')}, indent=2))


if __name__ == '__main__':
    main()
