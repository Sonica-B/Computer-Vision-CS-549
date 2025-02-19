import numpy as np
import cv2
from EstimateFundamentalMatrix import *

def InlierRANSAC(pts1, pts2, M=1000, epsilon=0.03):
    max_inliers = 0
    best_F = None
    best_inliers = None

    for _ in range(M):
        idx = np.random.choice(len(pts1), 8, replace=False)
        sample_pts1 = pts1[idx]
        sample_pts2 = pts2[idx]

        F, _ = cv2.findFundamentalMat(sample_pts1, sample_pts2, method=cv2.FM_8POINT)
        F = estimate_fundamental_matrix(sample_pts1, sample_pts2)

        inliers = []
        for i in range(len(pts1)):
            x1 = np.append(pts1[i], 1)
            x2 = np.append(pts2[i], 1)
            # print(x1, x2)
            error = np.abs(x2.T @ F @ x1)
            
            if error < epsilon:
                inliers.append(i)

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers