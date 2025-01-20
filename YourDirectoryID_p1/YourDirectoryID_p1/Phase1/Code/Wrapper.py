#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import argparse
import os

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=10000, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--folder', default='../Data/Train/Set1', help='Path to image folder')
    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    folder_path = Args.folder

    """
    Read a set of images for Panorama stitching
    """

    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)

    if len(images) < 3:
        print("Need at least 3 images")
        return

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    def detect_corners(img, Nstrong=1000):
        """
        Detect corners using Harris corner detector
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Find local maxima
        corners = []
        corner_scores = []
        for i in range(1, dst.shape[0] - 1):
            for j in range(1, dst.shape[1] - 1):
                if dst[i, j] > 0.01 * dst.max() and dst[i, j] == np.max(dst[i - 1:i + 2, j - 1:j + 2]):
                    corners.append([j, i])  # x, y coordinates
                    corner_scores.append(dst[i, j])

        corners = np.array(corners)
        corner_scores = np.array(corner_scores)

        return corners, corner_scores, dst

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    def adaptiveNonMaximalSuppression(corner_img, corners, N_best):
        """
        Implements ANMS to get well distributed corners
        """
        N_strong = len(corners)
        r = np.full(N_strong, np.inf)

        # Get y, x coordinates from corners
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        # Main ANMS loop
        for i in range(N_strong):
            for j in range(N_strong):
                # Check corner response
                if corner_img[int(y_coords[j]), int(x_coords[j])] > corner_img[int(y_coords[i]), int(x_coords[i])]:
                    # Calculate Euclidean distance
                    ED = (x_coords[j] - x_coords[i]) ** 2 + (y_coords[j] - y_coords[i]) ** 2
                    if ED < r[i]:
                        r[i] = ED

        # Sort r in descending order and pick top N_best points
        indices = np.argsort(r)[::-1]
        N_best = min(N_best, len(indices))
        selected_indices = indices[:N_best]

        return corners[selected_indices]
    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
    def extract_features(img, corners):
        """
        Extract feature descriptors
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        descriptors = []
        valid_corners = []

        for x, y in corners:
            # Extract 41x41 patch centered at corner
            x, y = int(x), int(y)
            if y - 20 < 0 or y + 21 > gray.shape[0] or x - 20 < 0 or x + 21 > gray.shape[1]:
                continue

            patch = gray[y - 20:y + 21, x - 20:x + 21]  # 41x41 patch

            # Apply Gaussian blur
            patch = cv2.GaussianBlur(patch, (3, 3), 0)

            # Subsample to 8x8
            patch = cv2.resize(patch, (8, 8))

            # Reshape to 64x1 and standardize
            feat = patch.reshape(64)
            feat = (feat - feat.mean()) / (feat.std() + 1e-7)

            descriptors.append(feat)
            valid_corners.append([x, y])

        return np.array(descriptors), np.array(valid_corners)

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    def match_features(desc1, desc2, corners1, corners2):
        """
        Match features using ratio test
        """
        matches = []

        for i, desc in enumerate(desc1):
            # Compute distances to all descriptors in desc2
            dists = np.sum((desc2 - desc) ** 2, axis=1)

            # Find best and second best matches
            idx = np.argsort(dists)
            if len(idx) < 2:
                continue

            best = dists[idx[0]]
            second_best = dists[idx[1]]

            # Apply ratio test
            if best < 0.8 * second_best:
                matches.append((corners1[i], corners2[idx[0]]))

        return np.array(matches)

    """
	Refine: RANSAC, Estimate Homography
	"""

    def ransac_homography(matches, threshold=5.0, max_iters=1000):
        """
        Estimate homography using RANSAC
        """
        if len(matches) < 4:
            return None, None

        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        best_H = None
        best_mask = None
        best_inliers = 0

        for _ in range(max_iters):
            # 1. Select 4 random correspondences
            idx = np.random.choice(len(matches), 4, replace=False)
            sample1 = pts1[idx]
            sample2 = pts2[idx]

            # 2. Compute homography
            H = cv2.findHomography(sample1.reshape(-1, 1, 2),
                                   sample2.reshape(-1, 1, 2))[0]
            if H is None:
                continue

            # 3. Count inliers
            pts1_h = np.hstack((pts1, np.ones((len(pts1), 1))))
            pts2_proj = (H @ pts1_h.T).T
            pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:]

            distances = np.linalg.norm(pts2 - pts2_proj, axis=1)
            mask = distances < threshold
            num_inliers = np.sum(mask)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_H = H
                best_mask = mask

            # Early termination if found 90% inliers
            if best_inliers > len(pts1) * 0.9:
                break

        # 6. Recompute H using all inliers
        if best_mask is not None and np.sum(best_mask) >= 4:
            best_H = cv2.findHomography(pts1[best_mask].reshape(-1, 1, 2),
                                        pts2[best_mask].reshape(-1, 1, 2))[0]

        return best_H, best_mask

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


    def blend_images(img1, img2, H):
        """
        Blend images using homography
        """
        # Find panorama dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)))

        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

        # Translation matrix
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        # Warp images
        warped1 = cv2.warpPerspective(img1, Ht @ H, (xmax - xmin, ymax - ymin))
        warped2 = cv2.warpPerspective(img2, Ht, (xmax - xmin, ymax - ymin))

        # Find overlap region
        mask1 = (cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY) > 0)
        mask2 = (cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0)
        overlap = mask1 & mask2

        # Alpha blending in overlap region
        alpha = 0.5
        for c in range(3):
            warped1[overlap, c] = (alpha * warped1[overlap, c] +
                                   (1 - alpha) * warped2[overlap, c]).astype(np.uint8)
        warped1[~mask1] = warped2[~mask1]

        return warped1


# Implementation

    # Corner Detection
    corners, corner_scores, corner_img = detect_corners(images[0])
    corner_vis = images[0].copy()
    for x, y in corners.astype(np.int32):
        cv2.circle(corner_vis, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite('corners.png', corner_vis)

    # ANMS
    anms_corners = adaptiveNonMaximalSuppression(corner_img, corners, NumFeatures)
    anms_vis = images[0].copy()
    for x, y in anms_corners.astype(np.int32):
        cv2.circle(anms_vis, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite('anms.png', anms_vis)

    # Feature Descriptors
    desc1, valid_corners1 = extract_features(images[0], anms_corners)
    fd_vis = images[0].copy()
    for x, y in valid_corners1.astype(np.int32):
        cv2.circle(fd_vis, (x, y), 3, (0, 255, 0), -1)
        cv2.rectangle(fd_vis, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 1)
    cv2.imwrite('FD.png', fd_vis)

    # Feature Matching
    corners2, corner_scores2, corner_img2 = detect_corners(images[1])
    anms_corners2 = adaptiveNonMaximalSuppression(corner_img2, corners2, NumFeatures)
    desc2, valid_corners2 = extract_features(images[1], anms_corners2)

    matches = match_features(desc1, desc2, valid_corners1, valid_corners2)

    # Visualize matches
    match_vis = np.zeros((max(images[0].shape[0], images[1].shape[0]),
                          images[0].shape[1] + images[1].shape[1], 3), dtype=np.uint8)
    match_vis[:images[0].shape[0], :images[0].shape[1]] = images[0]
    match_vis[:images[1].shape[0], images[0].shape[1]:] = images[1]

    for (x1, y1), (x2, y2) in matches:
        x2 = int(x2 + images[0].shape[1])
        cv2.line(match_vis, (int(x1), int(y1)), (x2, int(y2)), (0, 255, 0), 1)
    cv2.imwrite('matching.png', match_vis)

    # RANSAC Homography
    H, mask = ransac_homography(matches)

    # Image Blending
    if H is not None:
        panorama = blend_images(images[0], images[1], H)
        cv2.imwrite('mypano.png', panorama)
    else:
        print("Failed to compute homography")


if __name__ == "__main__":
    main()
