import numpy as np
from LinearTriangulation import LinearTriangulation


def DisambiguateCameraPose(K, C_set, R_set, x1, x2):
    """
    Disambiguate camera pose from the four possible configurations using cheirality condition

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        C_set (list): List of possible camera centers
        R_set (list): List of possible rotation matrices
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)

    Returns:
        tuple: Best camera center, rotation matrix, and reconstructed 3D points
    """
    # First camera is assumed to be at origin with identity rotation
    C1 = np.zeros((3, 1))
    R1 = np.eye(3)

    max_positive_depths = 0
    best_config = 0

    # Test all four configurations
    for i in range(len(C_set)):
        C2 = C_set[i].reshape(3, 1)
        R2 = R_set[i]

        # Triangulate points with current pose
        X = LinearTriangulation(K, C1, R1, C2, R2, x1, x2)

        # Count points with positive depth in both cameras
        positive_depths = 0

        # Check depth in first camera
        X_c1 = X - C1.reshape(1, 3)  # Points in camera 1 frame
        Z1 = (R1 @ X_c1.T)[2]  # Z coordinates in camera 1 frame

        # Check depth in second camera
        X_c2 = X - C2.reshape(1, 3)  # Points in camera 2 frame
        Z2 = (R2 @ X_c2.T)[2]  # Z coordinates in camera 2 frame

        # Count points with positive depth in both cameras
        positive_depths = np.sum((Z1 > 0) & (Z2 > 0))

        if positive_depths > max_positive_depths:
            max_positive_depths = positive_depths
            best_config = i
            best_X = X

    return C_set[best_config], R_set[best_config], best_X