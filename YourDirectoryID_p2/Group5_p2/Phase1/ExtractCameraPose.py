import numpy as np

# def ExtractCameraPoses(E):
#     W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#     U, D, Vt = np.linalg.svd(E)
#     cen = [U[:,2], -U[:,2], U[:,2], -U[:,2]]
#     rot = [U @ W @ Vt, U @ W @ Vt, U @ W.T @ Vt, U @ W.T @ Vt]
#     return cen, rot


import numpy as np


def ExtractCameraPoses(E):
    """
    Extract the four possible camera poses from the essential matrix
    """
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Ensure proper rotation with positive determinant
    if np.linalg.det(U @ W @ Vt) < 0:
        W = -W

    # Get two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Get two possible translations
    t = U[:, 2]

    # Four possible combinations
    C1 = t.reshape(3, 1)
    C2 = -t.reshape(3, 1)
    C3 = t.reshape(3, 1)
    C4 = -t.reshape(3, 1)

    R1 = np.where(np.linalg.det(R1) < 0, -R1, R1)
    R2 = np.where(np.linalg.det(R2) < 0, -R2, R2)

    cen = [C1, C2, C3, C4]
    rot = [R1, R1, R2, R2]

    return cen, rot