"""裁剪几何计算：只处理框和坐标，不依赖YOLO、GPU或视频解码器。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def interpolate_boxes(boxes, valid, times, max_gap_s=.5):
    """在同一人物轨迹内补齐短漏检，返回[T,4]原图框。

    不向首尾外推，也不跨长遮挡强行补框；插值后原始valid仍由调用方保存。
    """
    boxes, valid, times = np.asarray(boxes), np.asarray(valid, dtype=bool), np.asarray(times)
    if boxes.shape != (len(times), 4) or valid.shape != times.shape or len(times) < 2:
        raise ValueError('Expected at least two matching box/time/mask samples')
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any() or max_gap_s < 0:
        raise ValueError('Invalid timestamps or gap duration')
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError('No frames for selected track')
    if indices[0] != 0 or indices[-1] != len(times) - 1:
        raise ValueError('Selected track does not cover endpoints; choose a valid interval explicitly')
    if not np.isfinite(boxes[valid]).all() or (boxes[valid, 2:] <= boxes[valid, :2]).any():
        raise ValueError('Invalid observed boxes')
    # 只检查两个有效观测之间缺失了多久，不能把另一个人的框接进来填空。
    for left, right in zip(indices[:-1], indices[1:]):
        if right > left + 1 and times[right - 1] - times[left] > max_gap_s + 1e-8:
            raise ValueError('Selected track has a long gap; refusing identity bridging')
    return np.column_stack([np.interp(times, times[indices], boxes[indices, column]) for column in range(4)])


def plan_crops(boxes, width, height, fps=30, margin=1.3, smoothing_s=.15):
    """原图检测框[T,4] → 方形裁剪框[T,4]，坐标仍是原图像素。

    当前策略是整段固定边长、中心平滑跟随，避免随姿势变化不停放大缩小。
    会使用整段统计和未来帧，因此是离线预处理，不是实时因果算法。
    """
    if (width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0 or
            not np.isfinite(margin) or margin < 1.1 or not np.isfinite(smoothing_s) or smoothing_s < 0 or
            len(boxes) < 2 or not np.isfinite(boxes).all()):
        raise ValueError('Invalid crop parameters or boxes')
    sizes = boxes[:, 2:] - boxes[:, :2]
    if (sizes <= 0).any():
        raise ValueError('Bounding boxes must have positive sizes')
    # 用整段检测框的最大宽/高决定边长，margin=1.3表示边长扩大30%。
    # 向上取偶数像素，便于后续图像和视频编码处理。
    side = int(np.ceil(sizes.max() * margin / 2) * 2)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    if smoothing_s:
        # Gaussian的sigma以帧为单位；平滑秒数乘FPS即可换算。
        centers = gaussian_filter1d(centers, smoothing_s * fps, axis=0, mode='nearest')
    for axis, extent in enumerate((width, height)):
        if side <= extent:
            centers[:, axis] = np.clip(centers[:, axis], side / 2, extent - side / 2)
    # 优先保证检测框及每侧5%的宽高余量落在窗口内，而不是硬性禁止黑色补边。
    # 前面的平滑中心若会截掉手脚，就投影回当前检测框允许的中心区间。
    safe_lower = boxes[:, :2] - sizes * .05
    safe_upper = boxes[:, 2:] + sizes * .05
    centers = np.minimum(np.maximum(centers, safe_upper - side / 2), safe_lower + side / 2)
    origin = np.rint(centers - side / 2).astype(np.int32)
    return np.column_stack((origin, origin + side))


def crop_affines(crops):
    """计算逐帧2×3仿射矩阵，返回 原图→512图 和 512图→原图 两个方向。"""
    sizes = crops[:, 2:] - crops[:, :2]
    if (sizes <= 0).any() or not np.array_equal(sizes[:, 0], sizes[:, 1]):
        raise ValueError('Expected positive square crops')
    # 对原图坐标：x_crop = scale*(x_original-x0)，y同理。
    scale = 512 / sizes[:, 0]
    affine = np.zeros((len(crops), 2, 3), dtype=np.float64)
    affine[:, 0, 0], affine[:, 1, 1] = scale, scale
    affine[:, 0, 2], affine[:, 1, 2] = -crops[:, 0] * scale, -crops[:, 1] * scale
    # 保留逆变换，后续可把预测关键点投回原视频，而不必重新估计裁剪位置。
    inverse = np.zeros_like(affine)
    inverse[:, 0, 0], inverse[:, 1, 1] = 1 / scale, 1 / scale
    inverse[:, 0, 2], inverse[:, 1, 2] = crops[:, 0], crops[:, 1]
    return affine, inverse
