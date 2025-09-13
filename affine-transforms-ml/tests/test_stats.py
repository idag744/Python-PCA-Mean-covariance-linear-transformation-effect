import numpy as np
from src.stats import mean, covariance

def test_mean():
    X = np.array([[0., 1., 1.], [1., 2., 1.]])
    expected = np.array([0.5, 1.5, 1.])
    np.testing.assert_allclose(mean(X), expected)

def test_covariance():
    X = np.array([[0., 1.], [1., 2.], [0., 1.], [1., 2.]])
    expected = np.array([[0.25, 0.25], [0.25, 0.25]])
    np.testing.assert_allclose(covariance(X), expected)

