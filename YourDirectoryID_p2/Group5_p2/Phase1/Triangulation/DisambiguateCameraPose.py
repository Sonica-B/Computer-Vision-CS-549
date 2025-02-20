import numpy as np
from .LinearTriangulation import LinearTriangulation


def DisambiguateCameraPose(K, C_set, R_set, x1, x2):
    """
    Disambiguate camera pose from the four possible configurations using cheirality condition
    """
    # First camera is assumed to be at origin with identity rotation
    C1 = np.zeros((3, 1))
    R1 = np.eye(3)

    max_positive_depths = 0
    best_config = 0
    best_X = None

    # Test all four configurations
    for i in range(len(C_set)):
        C2 = C_set[i].reshape(3, 1)
        R2 = R_set[i]

        # Triangulate points with current pose
        X = LinearTriangulation(K, C1, R1, C2, R2, x1, x2)

        # Check depth in first camera
        X_c1 = X - C1.reshape(1, 3)
        Z1 = (R1 @ X_c1.T)[2]

        # Check depth in second camera
        X_c2 = X - C2.reshape(1, 3)
        Z2 = (R2 @ X_c2.T)[2]

        # Count points with positive depth in both cameras
        positive_depths = np.sum((Z1 > 0) & (Z2 > 0))

        # Update best configuration if current one is better
        if positive_depths > max_positive_depths:
            max_positive_depths = positive_depths
            best_config = i
            best_X = X

    print(f"Best configuration had {max_positive_depths} points in front of both cameras")
    if max_positive_depths < 10:
        print("WARNING: Very few points passed cheirality check!")

    return C_set[best_config], R_set[best_config], best_X