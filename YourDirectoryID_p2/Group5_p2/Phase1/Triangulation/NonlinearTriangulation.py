import numpy as np
from scipy.optimize import least_squares


def project_points(X, K, C, R):
    """
    Project 3D points to 2D using camera parameters
    """
    # Convert C to column vector if it's not
    C = C.reshape(3, 1)

    # Create projection matrix
    P = K @ np.hstack((R, -R @ C))

    # Convert to homogeneous coordinates
    X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))

    # Project points
    x_projected_homogeneous = (P @ X_homogeneous.T).T

    # Convert to inhomogeneous coordinates
    x_projected = x_projected_homogeneous[:, :2] / x_projected_homogeneous[:, 2:]

    return x_projected


def reprojection_error(X, K, C1, R1, C2, R2, x1, x2):
    """
    Calculate reprojection error for a single 3D point
    """
    # Project 3D point to both cameras
    x1_proj = project_points(X.reshape(1, 3), K, C1, R1)
    x2_proj = project_points(X.reshape(1, 3), K, C2, R2)

    # Calculate error
    error = np.concatenate([
        (x1_proj - x1).ravel(),
        (x2_proj - x2).ravel()
    ])

    return error


def NonlinearTriangulation(K, C1, R1, C2, R2, x1, x2, X_initial):
    """
    Refine 3D point locations using non-linear optimization

    Args:
        K (ndarray): Camera intrinsic matrix (3x3)
        C1 (ndarray): First camera center (3x1)
        R1 (ndarray): First camera rotation matrix (3x3)
        C2 (ndarray): Second camera center (3x1)
        R2 (ndarray): Second camera rotation matrix (3x3)
        x1 (ndarray): Points in first image (Nx2)
        x2 (ndarray): Points in second image (Nx2)
        X_initial (ndarray): Initial 3D points from linear triangulation (Nx3)

    Returns:
        ndarray: Refined 3D points (Nx3)
    """
    X_refined = np.zeros_like(X_initial)

    # Refine each point independently
    for i in range(len(X_initial)):
        # Define the optimization function
        def objective(X):
            return reprojection_error(X, K, C1, R1, C2, R2,
                                      x1[i].reshape(1, 2),
                                      x2[i].reshape(1, 2))

        # Optimize using least squares
        result = least_squares(objective, X_initial[i],
                               method='lm',  # Levenberg-Marquardt algorithm
                               max_nfev=200)  # Maximum number of function evaluations

        X_refined[i] = result.x

    return X_refined