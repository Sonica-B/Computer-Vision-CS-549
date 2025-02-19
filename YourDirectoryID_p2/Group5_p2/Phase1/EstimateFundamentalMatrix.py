import numpy as np
from scipy.linalg import svd

def estimate_fundamental_matrix(points1, points2):
    num_points = points1.shape[0]
    A = np.zeros((num_points, 9))
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
    U, S, V = svd(A)
    return V[8].reshape(3, 3)