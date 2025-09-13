import numpy as np

def mean(X: np.ndarray) -> np.ndarray:
    """
    Compute the sample mean of a dataset.

    Args:
        X (ndarray): Shape (N, D), where N = number of data points,
                     D = dimensionality of each data point.

    Returns:
        ndarray: Shape (D,), the sample mean.
    """
    return np.mean(X, axis=0)


def covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix of a dataset.

    Args:
        X (ndarray): Shape (N, D), dataset of N points in D dimensions.

    Returns:
        ndarray: Shape (D, D), covariance matrix.
    """
    X = np.array(X, dtype=float)
    N = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    return (X_centered.T @ X_centered) / (N - 1)

