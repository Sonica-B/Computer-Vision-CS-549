import numpy as np


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Triangulate 3D points from 2D correspondences using linear method

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        C1 (ndarray): First camera center (3x1)
        R1 (ndarray): First camera rotation matrix (3x3)
        C2 (ndarray): Second camera center (3x1)
        R2 (ndarray): Second camera rotation matrix (3x3)
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)

    Returns:
        ndarray: 3D points (Nx3)
    """
    # Convert points to homogeneous coordinates
    x1_homogeneous = np.column_stack((x1, np.ones(len(x1))))
    x2_homogeneous = np.column_stack((x2, np.ones(len(x2))))

    # Calculate projection matrices
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))

    # Initialize array for 3D points
    X = np.zeros((len(x1), 3))

    for i in range(len(x1)):
        # Build matrix A for current point pair
        A = np.zeros((4, 4))

        # First image constraints
        A[0] = x1_homogeneous[i, 0] * P1[2] - P1[0]
        A[1] = x1_homogeneous[i, 1] * P1[2] - P1[1]

        # Second image constraints
        A[2] = x2_homogeneous[i, 0] * P2[2] - P2[0]
        A[3] = x2_homogeneous[i, 1] * P2[2] - P2[1]

        # Solve for 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]

        # Convert to non-homogeneous coordinates
        X[i] = X_homogeneous[:3] / X_homogeneous[3]

    return X