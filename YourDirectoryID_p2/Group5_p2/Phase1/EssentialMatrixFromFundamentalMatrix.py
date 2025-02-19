import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    S_corrected = np.diag([1, 1, 0])
    E_corrected = U @ S_corrected @ Vt
    return E_corrected