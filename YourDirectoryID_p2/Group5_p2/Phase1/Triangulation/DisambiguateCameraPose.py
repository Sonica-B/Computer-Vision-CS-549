import numpy as np
from .LinearTriangulation import LinearTriangulation


def DisambiguateCameraPose(K, Cs, Rs, x1, x2):
    """
    Disambiguate camera pose from the four possible configurations using cheirality condition

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        Cs (list): List of possible camera centers
        Rs (list): List of possible rotation matrices
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)

    Returns:
        tuple: Best camera center, rotation matrix, and reconstructed 3D points
    """
    # First camera is at origin with identity rotation
    C1 = np.zeros(3)
    R1 = np.eye(3)

    max_positive_depths = 0
    best_config = 0
    best_X = None

    # Test all four configurations
    for i in range(len(Cs)):
        # Get current camera pose
        C2 = Cs[i]
        R2 = Rs[i]

        # Triangulate points
        X = LinearTriangulation(K, C1, R1, C2, R2, x1, x2)

        # Count points with positive depth in both cameras
        # For first camera
        X_c1 = X - C1  # Points in camera 1 frame
        Z1 = (R1 @ X_c1.T)[2]  # Z coordinates in camera 1 frame

        # For second camera
        X_c2 = X - C2.reshape(1, 3)  # Points in camera 2 frame
        Z2 = (R2 @ X_c2.T)[2]  # Z coordinates in camera 2 frame

        # Count points with positive depth in both cameras
        positive_depths = np.sum((Z1 > 0) & (Z2 > 0))

        print(f"Configuration {i + 1} has {positive_depths} points in front of both cameras")

        # Update best configuration if current is better
        if positive_depths > max_positive_depths:
            max_positive_depths = positive_depths
            best_config = i
            best_X = X

    print(f"Best configuration had {max_positive_depths} points in front of both cameras")
    if max_positive_depths < 10:
        print("WARNING: Very few points passed cheirality check!")

    return Cs[best_config], Rs[best_config], best_X