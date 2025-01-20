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
    Parser.add_argument('--NumFeatures', default=1000, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--folder', default='../Data/Train/Set2', help='Path to image folder')
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

    def detect_corners(img):
        """
        Detect corners using either Harris corners or Shi-Tomasi

        Args:
            img: Input image

        Returns:
            C_img: Corner response image from cornermetric
            corners: Detected corner coordinates
            scores: Corner response scores
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Method 1: Harris Corner Detector
        # Compute corner response image
        C_img = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Alternative Method 2: Shi-Tomasi
        # corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000,
        #                                  qualityLevel=0.01, minDistance=10)

        # Find local maxima
        corners = []
        scores = []

        # Pad the corner response image to handle boundaries
        padded = np.pad(C_img, ((1, 1), (1, 1)), mode='constant')

        # Find local maxima
        for i in range(1, C_img.shape[0] + 1):
            for j in range(1, C_img.shape[1] + 1):
                # 3x3 window for local maximum check
                window = padded[i - 1:i + 2, j - 1:j + 2]
                center = window[1, 1]

                # Check if center is local maximum and above threshold
                if center > 0.01 * C_img.max() and center == np.max(window):
                    # Note: x=j-1, y=i-1 to convert back to image coordinates
                    corners.append([j - 1, i - 1])
                    scores.append(center)

        return C_img, np.array(corners), np.array(scores)

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    def adaptiveNonMaximalSuppression(C_img, N_best):
        """
        Implement ANMS as specified in the algorithm

        Args:
            C_img: Corner score image (obtained using cornermetric)
            N_best: Number of best corners needed

        Returns:
            Array of (x_i, y_i) coordinates for i = 1:N_best
        """
        # Find all local maxima using imregionalmax
        local_maxima = []
        scores = []

        # Pad the corner response image
        padded = np.pad(C_img, ((1, 1), (1, 1)), mode='constant')

        # Find all local maxima
        for i in range(1, C_img.shape[0] + 1):
            for j in range(1, C_img.shape[1] + 1):
                window = padded[i - 1:i + 2, j - 1:j + 2]
                center = window[1, 1]

                if center == np.max(window) and center > 0:
                    # Store [j-1, i-1] as x=j-1, y=i-1 are the image coordinates
                    local_maxima.append([j - 1, i - 1])
                    scores.append(center)

        local_maxima = np.array(local_maxima)
        N_strong = len(local_maxima)

        if N_strong == 0:
            return np.array([])

        # Initialize r_i = infinity for i = [1:N_strong]
        r = np.full(N_strong, np.inf)

        # For each corner
        for i in range(N_strong):
            # Get coordinates of current corner
            y_i, x_i = local_maxima[i]  # Note: these are already in image coordinates

            # Compare with all other corners
            for j in range(N_strong):
                y_j, x_j = local_maxima[j]

                # Check if corner j has higher corner response
                if C_img[y_j, x_j] > C_img[y_i, x_i]:
                    # Calculate Euclidean distance
                    ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2

                    # Update r_i if necessary
                    if ED < r[i]:
                        r[i] = ED

        # Sort r_i in descending order and pick top N_best points
        idx = np.argsort(r)[::-1]  # Sort in descending order
        idx = idx[:min(N_best, len(idx))]  # Take top N_best points

        # Return coordinates in (x_i, y_i) format
        return np.array([[local_maxima[i][1], local_maxima[i][0]] for i in idx])

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

    def ransac_homography(matches, threshold=5.0, N_max=1000):
        """
        Estimate homography using RANSAC following the specified steps:
        1. Select four feature pairs (at random)
        2. Compute homography H between these pairs
        3. Compute inliers using SSD < τ
        4. Repeat until N_max iterations or >90% inliers found
        5. Keep largest set of inliers
        6. Re-compute least-squares H estimate on all inliers

        Args:
            matches: Nx2x2 array of matching point pairs
            threshold: τ for SSD threshold
            N_max: Maximum number of iterations

        Returns:
            H: 3x3 homography matrix
            mask: Boolean array indicating inliers
        """
        if len(matches) < 4:
            return None, None

        # Convert matches to separate source and destination points
        p = np.float32(matches[:, 0])  # p_i from image 1
        p_prime = np.float32(matches[:, 1])  # p'_i from image 2

        best_H = None
        best_mask = None
        best_inliers = 0

        for _ in range(N_max):
            # 1. Select four feature pairs at random
            idx = np.random.choice(len(matches), 4, replace=False)
            src_pts = p[idx].reshape(-1, 1, 2)
            dst_pts = p_prime[idx].reshape(-1, 1, 2)

            # 2. Compute homography H between point pairs
            H, _ = cv2.findHomography(src_pts, dst_pts)
            if H is None:
                continue

            # 3. Compute SSD for all points
            p_transformed = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H)
            if p_transformed is None:
                continue

            # Calculate SSD (sum of squared differences)
            ssd = np.sum((p_prime.reshape(-1, 1, 2) - p_transformed) ** 2, axis=(1, 2))

            # Find inliers where SSD < τ
            mask = ssd < threshold
            num_inliers = np.sum(mask)

            # Update best result if we found more inliers
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_H = H
                best_mask = mask

            # 4. Early termination if we found more than 90% inliers
            if best_inliers > len(p) * 0.9:
                break

        # 5. Keep largest set of inliers
        if best_mask is None or np.sum(best_mask) < 4:
            return None, None

        # 6. Re-compute least-squares H estimate on all inliers
        src_pts_inliers = p[best_mask].reshape(-1, 1, 2)
        dst_pts_inliers = p_prime[best_mask].reshape(-1, 1, 2)
        best_H, _ = cv2.findHomography(src_pts_inliers, dst_pts_inliers, method=cv2.RANSAC)

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

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    # Detect corners in first image
    C_img, corners, corner_scores = detect_corners(images[0])

    # Visualize corners
    corner_vis = images[0].copy()
    for corner in corners:
        x, y = corner[0], corner[1]  # Correctly unpack x,y coordinates
        cv2.circle(corner_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite('corners.png', corner_vis)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    # Apply ANMS
    anms_corners = adaptiveNonMaximalSuppression(C_img, NumFeatures)

    # Visualize ANMS corners
    anms_vis = images[0].copy()
    for corner in anms_corners:
        x, y = corner[0], corner[1]  # Correctly unpack x,y coordinates
        cv2.circle(anms_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite('anms.png', anms_vis)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    # Extract features for ANMS corners
    desc1, valid_corners1 = extract_features(images[0], anms_corners)

    # Visualize feature patches
    fd_vis = images[0].copy()
    for corner in valid_corners1:
        x, y = corner[0], corner[1]  # Correctly unpack x,y coordinates
        cv2.circle(fd_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.rectangle(fd_vis, (int(x - 20), int(y - 20)), (int(x + 20), int(y + 20)), (0, 255, 0), 1)
    cv2.imwrite('FD.png', fd_vis)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    # Detect and process corners in second image
    C_img2, corners2, corner_scores2 = detect_corners(images[1])
    anms_corners2 = adaptiveNonMaximalSuppression(C_img2, NumFeatures)
    desc2, valid_corners2 = extract_features(images[1], anms_corners2)

    # Match features
    matches = match_features(desc1, desc2, valid_corners1, valid_corners2)

    # Visualize matches
    if len(matches) > 0:
        # Create match visualization
        match_vis = np.zeros((max(images[0].shape[0], images[1].shape[0]),
                              images[0].shape[1] + images[1].shape[1], 3), dtype=np.uint8)
        match_vis[:images[0].shape[0], :images[0].shape[1]] = images[0]
        match_vis[:images[1].shape[0], images[0].shape[1]:] = images[1]

        # Draw matches
        for (x1, y1), (x2, y2) in matches:
            x2 = int(x2 + images[0].shape[1])  # Offset x2 by width of first image
            cv2.line(match_vis, (int(x1), int(y1)), (x2, int(y2)), (0, 255, 0), 1)
        cv2.imwrite('matching.png', match_vis)

    """
    Refine: RANSAC, Estimate Homography
    """
    if len(matches) > 0:
        H, mask = ransac_homography(matches)

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        if H is not None:
            panorama = blend_images(images[0], images[1], H)
            cv2.imwrite('mypano.png', panorama)
        else:
            print("Failed to compute homography")
    else:
        print("No matches found between images")


if __name__ == "__main__":
    main()
