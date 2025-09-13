import numpy as np
from src.transforms import affine_mean, affine_covariance

def test_affine_mean_and_cov():
    A = np.array([[0, 1], [2, 3]])
    b = np.ones(2)
    m = np.full((2,), 2)
    S = np.eye(2) * 2

    expected_mean = np.array([3., 11.])
    expected_cov = np.array([[2., 6.], [6., 26.]])

    np.testing.assert_allclose(affine_mean(m, A, b), expected_mean)
    np.testing.assert_allclose(affine_covariance(S, A, b), expected_cov)

