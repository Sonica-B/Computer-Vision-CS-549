import numpy as np
from scipy.optimize import least_squares


def project_points(X, K, C, R):
    """Project 3D points to 2D"""
    # Ensure C is column vector
    C = C.reshape(3, 1)

    # Create projection matrix
    P = K @ np.hstack((R, -R @ C))

    # Convert to homogeneous coordinates
    X_homog = np.hstack((X, np.ones((X.shape[0], 1))))

    # Project
    x_proj_homog = (P @ X_homog.T).T
    x_proj = x_proj_homog[:, :2] / x_proj_homog[:, 2:]

    return x_proj


def reprojection_error(X, K, C1, R1, C2, R2, x1, x2):
    """Calculate reprojection error"""
    # Project points
    x1_proj = project_points(X.reshape(-1, 3), K, C1, R1)
    x2_proj = project_points(X.reshape(-1, 3), K, C2, R2)

    # Calculate error
    error = np.concatenate([
        (x1_proj - x1).ravel(),
        (x2_proj - x2).ravel()
    ])

    return error


def NonlinearTriangulation(K, C1, R1, C2, R2, x1, x2, X_init):
    """
    Refine 3D point locations using non-linear optimization

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        C1 (ndarray): First camera center (3,)
        R1 (ndarray): First camera rotation matrix (3x3)
        C2 (ndarray): Second camera center (3,)
        R2 (ndarray): Second camera rotation matrix (3x3)
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)
        X_init (ndarray): Initial 3D points from linear triangulation (Nx3)

    Returns:
        ndarray: Refined 3D points (Nx3)
    """
    X_refined = np.zeros_like(X_init)

    # Refine each point independently
    for i in range(len(X_init)):
        # Define objective function
        def objective(X):
            return reprojection_error(X, K, C1, R1, C2, R2,
                                      x1[i:i + 1], x2[i:i + 1])

        # Optimize
        result = least_squares(objective, X_init[i],
                               method='lm',
                               max_nfev=200)

        X_refined[i] = result.x

    return X_refined