import numpy as np
from scipy.spatial.transform import Rotation

from data_pipeline.sources.smplx_dance import rotation_6d_to_matrix


def test_finedance_6d_uses_first_two_rows():
    expected = Rotation.from_euler('xyz', [30, -40, 80], degrees=True).as_matrix()
    six = expected[:2, :].reshape(6)
    np.testing.assert_allclose(rotation_6d_to_matrix(six), expected, atol=1e-12)
