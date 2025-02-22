import numpy as np


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Triangulate 3D points from 2D correspondences using linear method

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        C1 (ndarray): First camera center (3,)
        R1 (ndarray): First camera rotation matrix (3x3)
        C2 (ndarray): Second camera center (3,)
        R2 (ndarray): Second camera rotation matrix (3x3)
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)

    Returns:
        ndarray: 3D points (Nx3)
    """
    # Ensure C1 and C2 are column vectors
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)

    # Compute projection matrices
    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))

    # Number of points
    num_points = x1.shape[0]
    X = np.zeros((num_points, 3))

    for i in range(num_points):
        # Build the measurement matrix A
        A = np.zeros((4, 4))

        # For first image
        A[0] = x1[i, 0] * P1[2] - P1[0]
        A[1] = x1[i, 1] * P1[2] - P1[1]

        # For second image
        A[2] = x2[i, 0] * P2[2] - P2[0]
        A[3] = x2[i, 1] * P2[2] - P2[1]

        # Solve AX = 0 using SVD
        _, _, Vh = np.linalg.svd(A)
        X_homogeneous = Vh[-1]

        # Convert from homogeneous coordinates
        X[i] = X_homogeneous[:3] / X_homogeneous[3]

    return X

def normalize_points(points):
    """Normalize points to have zero mean and unit std"""
    mean = np.mean(points, axis=0)
    std = np.std(points)
    normalized_points = (points - mean) / std
    return normalized_points, mean, std

def denormalize_points(points, mean, std):
    """Denormalize points back to original scale"""
    return points * std + mean