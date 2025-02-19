import numpy as np

def ExtractCameraPoses(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, D, Vt = np.linalg.svd(E)
    cen = [U[:,2], -U[:,2], U[:,2], -U[:,2]]
    rot = [U @ W @ Vt, U @ W @ Vt, U @ W.T @ Vt, U @ W.T @ Vt]
    return cen, rot
