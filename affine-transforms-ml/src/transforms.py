import numpy as np

def affine_mean(mean: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the mean after affine transformation y = Ax + b.

    Args:
        mean (ndarray): Original mean (D,).
        A (ndarray): Transformation matrix (D, D).
        b (ndarray): Translation vector (D,).

    Returns:
        ndarray: Transformed mean (D,).
    """
    return A @ mean + b


def affine_covariance(S: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the covariance after affine transformation y = Ax + b.

    Args:
        S (ndarray): Original covariance matrix (D, D).
        A (ndarray): Transformation matrix (D, D).
        b (ndarray): Translation vector (ignored).

    Returns:
        ndarray: Transformed covariance (D, D).
    """
    return A @ S @ A.T

