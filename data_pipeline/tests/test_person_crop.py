import numpy as np
import pytest

from data_pipeline.video.geometry import interpolate_boxes, plan_crops


def test_short_gap_and_no_long_bridging():
    times = np.arange(30) / 30
    boxes = np.tile([100, 100, 200, 400], (30, 1)).astype(float)
    valid = np.ones(30, bool)
    valid[10:12] = False
    boxes[~valid] = np.nan
    assert np.isfinite(interpolate_boxes(boxes, valid, times)).all()
    valid[3:26] = False
    with pytest.raises(ValueError, match='long gap'):
        interpolate_boxes(boxes, valid, times)


def test_endpoints_must_not_be_extrapolated():
    with pytest.raises(ValueError, match='endpoints'):
        interpolate_boxes(np.ones((3, 4)), np.array([False, True, True]), np.arange(3) / 30)


def test_motion_crop_contains_person_and_has_constant_scale():
    x = np.linspace(20, 1600, 100)
    boxes = np.column_stack((x, x * 0 + 200, x + 180, x * 0 + 700))
    crops = plan_crops(boxes, 1920, 1080)
    assert ((boxes[:, :2] >= crops[:, :2]) & (boxes[:, 2:] <= crops[:, 2:])).all()
    assert len(np.unique(crops[:, 2] - crops[:, 0])) == 1
    assert np.ptp(crops[:, 0]) > 1000
