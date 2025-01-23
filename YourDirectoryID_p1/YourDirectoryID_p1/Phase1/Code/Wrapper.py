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
import torch
import argparse
import os

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=1000, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--folder', default='../Data/Train/Set3', help='Path to image folder')
    Parser.add_argument('--output_dir', default='Outputs', help='Output directory')
    Args = Parser.parse_args()

    os.makedirs(Args.output_dir, exist_ok=True)
    num_features = int(Args.NumFeatures)

    """
    Read a set of images for Panorama stitching
    """
    # Read images
    images = []
    for f in sorted(os.listdir(Args.folder)):
        if f.endswith('.jpg'):
            img = cv2.imread(os.path.join(Args.folder, f))
            if img is not None:
                images.append(img)

    if len(images) < 2:
        print("Need at least 2 images")
        return

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    def detect_corners(images, num_features):
        all_corners = []
        all_responses = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            response = cv2.dilate(response, None)

            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=num_features * 2,
                qualityLevel=0.01,
                minDistance=20,
                useHarrisDetector=True,
                k=0.04
            )

            if corners is not None:
                corners = corners.reshape(-1, 2)
                all_corners.append(corners)
                all_responses.append(response)

        return all_corners, all_responses
    # def detect_corners(img, use_gpu=False):
    #
    #     if use_gpu:
    #         # Convert to grayscale
    #         gray_gpu = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #         # Create Harris detector
    #         harris = cv2.cuda.createHarrisCorner(
    #             blockSize=2,
    #             ksize=3,
    #             k=0.04
    #         )
    #
    #         # Compute corner response
    #         C_img_gpu = harris.compute(gray_gpu)
    #         C_img = C_img_gpu.download()
    #     else:
    #
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         gray = np.float32(gray)
    #         C_img = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    #
    #     print("Finding local maxima...")
    #     # Find local maxima
    #     corners = []
    #     scores = []
    #
    #     for i in range(1, C_img.shape[0] - 1):
    #         for j in range(1, C_img.shape[1] - 1):
    #             window = C_img[i - 1:i + 2, j - 1:j + 2]
    #             center = C_img[i, j]
    #
    #             if center > 0.01 * C_img.max() and center == np.max(window):
    #                 corners.append([j, i])  # x=j, y=i
    #                 scores.append(center)
    #
    #     return C_img, np.array(corners), np.array(scores)

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    def adaptive_non_maximal_suppression(C_img, corners, N_best):
        N_strong = len(corners)
        if N_strong <= N_best:
            return corners

        r = np.full(N_strong, np.inf)
        scores = np.array([C_img[int(y), int(x)] for x, y in corners])

        # ANMS per project PDF
        for i in range(N_strong):
            for j in range(N_strong):
                if scores[j] > scores[i]:
                    ED = np.sum((corners[j] - corners[i]) ** 2)
                    r[i] = min(r[i], ED)

        idx = np.argsort(r)[::-1][:N_best]
        return corners[idx]

    # def adaptiveNonMaximalSuppression(C_img, N_best):
    #
    #     # Find all local maxima
    #     h, w = C_img.shape
    #     local_maxima = []
    #     scores = []
    #
    #     for i in range(1, h - 1):
    #         for j in range(1, w - 1):
    #             # Get 3x3 window
    #             window = C_img[i - 1:i + 2, j - 1:j + 2]
    #             center = C_img[i, j]
    #
    #             # Check if local maximum
    #             if center == np.max(window) and center > 0:
    #                 # Store [j,i] as [x,y] coordinates
    #                 local_maxima.append([j, i])  # x=j, y=i
    #                 scores.append(center)
    #
    #     local_maxima = np.array(local_maxima)
    #     N_strong = len(local_maxima)
    #
    #     if N_strong < 4:
    #         return local_maxima
    #
    #     # Initialize r_i = infinity for i = [1:N_strong]
    #     r = np.full(N_strong, np.inf)
    #
    #     # For each corner
    #     for i in range(N_strong):
    #         x_i, y_i = local_maxima[i]
    #
    #         # Compare with all other corners
    #         for j in range(N_strong):
    #             if i == j:  # Skip self-comparison
    #                 continue
    #
    #             x_j, y_j = local_maxima[j]
    #
    #             # Compare corner responses
    #             if C_img[y_j, x_j] > C_img[y_i, x_i]:
    #                 # Euclidean distance
    #                 ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
    #                 # Update r_i
    #                 r[i] = min(r[i], ED)
    #
    #     # Pick top N_best points
    #     idx = np.argsort(r)[::-1]
    #     idx = idx[:min(N_best, N_strong)]
    #
    #     return local_maxima[idx] # Array of (x_i, y_i) coordinates for i = 1:N_best

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    def extract_features(img, corners):
        descriptors = []
        valid_corners = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        for x, y in corners:
            x, y = int(x), int(y)
            if (y - 20 < 0 or y + 21 > img.shape[0] or
                    x - 20 < 0 or x + 21 > img.shape[1]):
                continue

            # Extract 41x41 patch per PDF
            patch = gray[y - 20:y + 21, x - 20:x + 21].copy()
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
            patch = cv2.resize(patch, (8, 8))

            feat = patch.reshape(64)
            feat = (feat - feat.mean()) / (feat.std() + 1e-7)

            descriptors.append(feat)
            valid_corners.append([x, y])

        return np.array(descriptors), np.array(valid_corners)
    # def extract_features(img, corners, use_gpu=False):
    #
    #     if use_gpu:
    #         gray_gpu = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         gray = gray_gpu.download()  # Need CPU for patch extraction
    #     else:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     descriptors = []
    #     valid_corners = []
    #     total = len(corners)
    #
    #     print(f"Extracting features from {total} corners...")
    #     for idx, (x, y) in enumerate(corners):
    #         if idx % 100 == 0:  # Progress update every 100 corners
    #             print(f"Processing corner {idx}/{total}")
    #
    #         x, y = int(x), int(y)
    #         # Check bounds for 41x41 patch
    #         if y - 20 < 0 or y + 21 > gray.shape[0] or x - 20 < 0 or x + 21 > gray.shape[1]:
    #             continue
    #
    #         # Extract patch
    #         patch = gray[y - 20:y + 21, x - 20:x + 21]
    #
    #         if use_gpu:
    #
    #             patch_gpu = cv2.cuda_GpuMat()
    #             patch_gpu.upload(patch)
    #
    #             # Gaussian blur
    #             patch_blur_gpu = cv2.cuda.createGaussianFilter(
    #                 cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0
    #             )
    #             patch_blur = patch_blur_gpu.apply(patch_gpu)
    #             patch = patch_blur.download()
    #         else:
    #             patch = cv2.GaussianBlur(patch, (3, 3), 0)
    #
    #         # Resize and normalize
    #         patch = cv2.resize(patch, (8, 8))
    #         feat = patch.reshape(64)
    #         feat = (feat - feat.mean()) / (feat.std() + 1e-7)
    #
    #         descriptors.append(feat)
    #         valid_corners.append([x, y])
    #
    #     return np.array(descriptors), np.array(valid_corners)

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    def match_features(desc1, desc2, corners1, corners2):
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

        if len(desc1) < 4 or len(desc2) < 4:
            return np.array([])

        bf = cv2.BFMatcher()
        matches = []

        try:
            # Forward matching
            matches12 = bf.knnMatch(desc1, desc2, k=2)
            matches21 = bf.knnMatch(desc2, desc1, k=2)

            # Ratio test + cross check
            forward = {}
            ratio = 0.8
            for m, n in matches12:
                if m.distance < ratio * n.distance:
                    forward[m.queryIdx] = m.trainIdx

            for m, n in matches21:
                if m.distance < ratio * n.distance:
                    if m.trainIdx in forward and forward[m.trainIdx] == m.queryIdx:
                        matches.append((corners1[m.trainIdx], corners2[m.queryIdx]))

        except cv2.error:
            return np.array([])

        return np.array(matches)

    # def match_features(desc1, desc2, corners1, corners2, use_gpu=False):
    #
    #     matches = []
    #     total = len(desc1)
    #
    #     print(f"Matching features between {total} and {len(desc2)} descriptors...")
    #     for i, desc in enumerate(desc1):
    #         if i % 100 == 0:  # Progress update
    #             print(f"Matching feature {i}/{total}")
    #
    #         if use_gpu:
    #             # Convert to GPU matrices
    #             desc_gpu = cv2.cuda_GpuMat()
    #             desc2_gpu = cv2.cuda_GpuMat()
    #             desc_gpu.upload(desc.reshape(1, -1))
    #             desc2_gpu.upload(desc2)
    #
    #             # Compute distances
    #             bf_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
    #             matches_gpu = bf_matcher.match(desc_gpu, desc2_gpu)
    #
    #             # Get best matches
    #             matches_gpu = sorted(matches_gpu, key=lambda x: x.distance)
    #             if len(matches_gpu) >= 2:
    #                 best = matches_gpu[0].distance
    #                 second_best = matches_gpu[1].distance
    #
    #                 if best < 0.8 * second_best:
    #                     matches.append((corners1[i], corners2[matches_gpu[0].trainIdx]))
    #         else:
    #             # CPU implementation
    #             dists = np.sum((desc2 - desc) ** 2, axis=1)
    #             idx = np.argsort(dists)
    #
    #             if len(idx) >= 2:
    #                 best = dists[idx[0]]
    #                 second_best = dists[idx[1]]
    #
    #                 if best < 0.8 * second_best:
    #                     matches.append((corners1[i], corners2[idx[0]]))
    #
    #     return np.array(matches)
    """
	Refine: RANSAC, Estimate Homography
	"""

    # def compute_homography_dlt(pts1, pts2):
    #     """Compute homography using Direct Linear Transform"""
    #     if len(pts1) < 4:
    #         return None
    #
    #     # Normalize points
    #     mean1 = np.mean(pts1, axis=0)
    #     mean2 = np.mean(pts2, axis=0)
    #     scale1 = np.sqrt(2) / np.std(pts1 - mean1)
    #     scale2 = np.sqrt(2) / np.std(pts2 - mean2)
    #
    #     T1 = np.array([
    #         [scale1, 0, -scale1 * mean1[0]],
    #         [0, scale1, -scale1 * mean1[1]],
    #         [0, 0, 1]
    #     ])
    #
    #     T2 = np.array([
    #         [scale2, 0, -scale2 * mean2[0]],
    #         [0, scale2, -scale2 * mean2[1]],
    #         [0, 0, 1]
    #     ])
    #
    #     # Transform points
    #     n = len(pts1)
    #     A = np.zeros((2 * n, 9))
    #
    #     # Normalized points
    #     pts1_norm = (T1 @ np.vstack((pts1.T, np.ones(n)))).T[:, :2]
    #     pts2_norm = (T2 @ np.vstack((pts2.T, np.ones(n)))).T[:, :2]
    #
    #     for i in range(n):
    #         x1, y1 = pts1_norm[i]
    #         x2, y2 = pts2_norm[i]
    #
    #         A[2 * i] = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]
    #         A[2 * i + 1] = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]
    #
    #     # SVD solution
    #     _, _, Vh = np.linalg.svd(A)
    #     H = Vh[-1].reshape(3, 3)
    #
    #     # Denormalize
    #     H = np.linalg.inv(T2) @ H @ T1
    #     return H / H[2, 2]

    def ransac_homography(matches, threshold=4.0, max_iters=2000):
        if len(matches) < 4:
            return None, None

        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        best_H = None
        best_mask = None
        best_inliers = 0
        min_inliers = 8

        for _ in range(max_iters):
            # Select 4 random pairs
            idx = np.random.choice(len(matches), 4, replace=False)
            src = pts1[idx].reshape(-1, 1, 2).astype(np.float32)
            dst = pts2[idx].reshape(-1, 1, 2).astype(np.float32)

            H = cv2.getPerspectiveTransform(src, dst)
            if H is None:
                continue

            # Transform points
            transformed = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2).astype(np.float32), H)
            if transformed is None:
                continue

            # Compute error
            errors = np.sqrt(np.sum((pts2.reshape(-1, 1, 2) - transformed) ** 2, axis=(1, 2)))
            mask = errors < threshold
            inliers = np.sum(mask)

            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
                best_mask = mask

                if best_inliers > len(matches) * 0.6:
                    break

        if best_inliers < min_inliers:
            return None, None

        return best_H, best_mask

    # def check_homography_quality(H, matches, threshold=5.0):
    #
    #     if H is None:
    #         return False
    #
    #     # Check determinant
    #     det = np.linalg.det(H)
    #     if det < 0 or abs(det - 1) > 0.1:
    #         return False
    #
    #     # Check condition number
    #     if np.linalg.cond(H) > 10000:
    #         return False
    #
    #     # Check reprojection error
    #     pts1 = matches[:, 0].reshape(-1, 1, 2).astype(np.float32)
    #     pts2 = matches[:, 1].reshape(-1, 1, 2).astype(np.float32)
    #
    #     transformed = cv2.perspectiveTransform(pts1, H)
    #     errors = np.sqrt(np.sum((pts2 - transformed) ** 2, axis=2)).ravel()
    #
    #     return np.mean(errors) < threshold

    # def ransac_homography(matches, threshold=5.0, max_iters=1000, use_gpu=False):
    #
    #     if len(matches) < 4:
    #         return None, None
    #
    #     p = np.float32(matches[:, 0])
    #     p_prime = np.float32(matches[:, 1])
    #
    #     best_H = None
    #     best_mask = None
    #     best_inliers = 0
    #
    #     print(f"Running RANSAC for max {max_iters} iterations...")
    #     for iter in range(max_iters):
    #         if iter % 100 == 0:  # Progress update
    #             print(f"RANSAC iteration {iter}/{max_iters}")
    #
    #         # Select random points
    #         idx = np.random.choice(len(matches), 4, replace=False)
    #         src_pts = p[idx].reshape(-1, 1, 2)
    #         dst_pts = p_prime[idx].reshape(-1, 1, 2)
    #
    #         if use_gpu:
    #
    #             src_gpu = cv2.cuda_GpuMat()
    #             dst_gpu = cv2.cuda_GpuMat()
    #             src_gpu.upload(src_pts)
    #             dst_gpu.upload(dst_pts)
    #
    #
    #             H_gpu = cv2.cuda.findHomography(src_gpu, dst_gpu)
    #             H = H_gpu.download()
    #         else:
    #             H = cv2.findHomography(src_pts, dst_pts)[0]
    #
    #         if H is None:
    #             continue
    #
    #         # Transform points
    #         if use_gpu:
    #             p_gpu = cv2.cuda_GpuMat()
    #             p_gpu.upload(p.reshape(-1, 1, 2))
    #             transformed_gpu = cv2.cuda.warpPerspective(p_gpu, H_gpu, p.shape[::-1])
    #             transformed = transformed_gpu.download()
    #         else:
    #             transformed = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H)
    #
    #         if transformed is None:
    #             continue
    #
    #         # Compute inliers
    #         ssd = np.sum((p_prime.reshape(-1, 1, 2) - transformed) ** 2, axis=(1, 2))
    #         mask = ssd < threshold
    #         inliers = np.sum(mask)
    #
    #         if inliers > best_inliers:
    #             best_inliers = inliers
    #             best_H = H
    #             best_mask = mask
    #             print(f"Found better model with {inliers} inliers")
    #
    #         if best_inliers > len(p) * 0.9:
    #             print("Early termination - found good model")
    #             break
    #
    #     print(f"RANSAC completed with {best_inliers} inliers")
    #     return best_H, best_mask

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    def blend_images(img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners2 = corners2.reshape(-1, 2)
        img2_corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        all_corners = np.vstack((corners2, img2_corners))

        xmin, ymin = np.int32(np.min(all_corners, axis=0).ravel() - 10)
        xmax, ymax = np.int32(np.max(all_corners, axis=0).ravel() + 10)

        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
        H_final = translation @ H
        output_shape = (xmax - xmin, ymax - ymin)

        warped1 = cv2.warpPerspective(img1, H_final, output_shape)
        warped2 = cv2.warpPerspective(img2, translation, output_shape)

        mask1 = cv2.warpPerspective(np.ones_like(img1[:, :, 0]), H_final, output_shape)
        mask2 = cv2.warpPerspective(np.ones_like(img2[:, :, 0]), translation, output_shape)

        # Simple alpha blending
        weight1 = mask1 / (mask1 + mask2 + 1e-10)
        weight2 = mask2 / (mask1 + mask2 + 1e-10)

        blended = warped1 * weight1[..., np.newaxis] + warped2 * weight2[..., np.newaxis]
        result = blended.astype(np.uint8)

        return result

    def create_panorama(images, num_features=1000):
        # Use first image as anchor
        panorama = images[0]

        for i in range(1, len(images)):
            # Double the features for better matching
            corners, responses = detect_corners([panorama, images[i]], num_features * 2)

            if not corners or len(corners) < 2:
                print(f"No corners detected in image {i}")
                continue

            # Get more corners than default
            all_anms_corners = []
            for corners_i, response_i in zip(corners, responses):
                anms_corners = adaptive_non_maximal_suppression(response_i, corners_i, num_features * 2)
                all_anms_corners.append(anms_corners)

            descriptors = []
            valid_corners = []
            for j, img in enumerate([panorama, images[i]]):
                desc, v_corners = extract_features(img, all_anms_corners[j])
                descriptors.append(desc)
                valid_corners.append(v_corners)

            matches = match_features(descriptors[0], descriptors[1], valid_corners[0], valid_corners[1])

            # Lower threshold for matches
            if len(matches) < 8:
                print(f"Only {len(matches)} matches found for image {i}")
                continue

            H, mask = ransac_homography(matches, threshold=5.0, max_iters=3000)
            if H is None:
                print(f"Failed to find homography for image {i}")
                continue

            panorama = blend_images(panorama, images[i], H)

        return panorama

    # def blend_images(img1, img2, H):
    #
    #     H = H.astype(np.float32)
    #
    #
    #     h1, w1 = img1.shape[:2]
    #     h2, w2 = img2.shape[:2]
    #
    #     # Transform corners
    #     corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    #     corners2 = cv2.perspectiveTransform(corners1, H)
    #     corners = np.concatenate((corners2, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)))
    #
    #     [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    #     [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    #
    #     # Translation matrix
    #     t = [-xmin, -ymin]
    #     Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)
    #
    #     # Warp images
    #     warped1 = cv2.warpPerspective(img1, Ht @ H, (xmax - xmin, ymax - ymin))
    #     warped2 = cv2.warpPerspective(img2, Ht, (xmax - xmin, ymax - ymin))
    #
    #     # Create masks
    #     mask1 = cv2.warpPerspective(np.ones_like(img1[:, :, 0]), Ht @ H, (xmax - xmin, ymax - ymin))
    #     mask2 = cv2.warpPerspective(np.ones_like(img2[:, :, 0]), Ht, (xmax - xmin, ymax - ymin))
    #
    #     # Blend images
    #     result = np.zeros_like(warped1, dtype=np.float32)
    #
    #     # Non-overlapping regions
    #     result[mask1 > 0] = warped1[mask1 > 0]
    #     result[mask2 > 0] = warped2[mask2 > 0]
    #
    #     # Overlapping regions
    #     overlap = (mask1 > 0) & (mask2 > 0)
    #     if overlap.any():
    #         alpha = 0.5
    #         result[overlap] = (alpha * warped1[overlap] + (1 - alpha) * warped2[overlap])
    #
    #     return result.astype(np.uint8)

    # def create_matching_visualization(img1, img2, matches, mask=None):
    #
    #     h1, w1 = img1.shape[:2]
    #     h2, w2 = img2.shape[:2]
    #
    #     vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    #     vis[:h1, :w1] = img1
    #     vis[:h2, w1:w1 + w2] = img2
    #
    #     for idx, ((x1, y1), (x2, y2)) in enumerate(matches):
    #         color = (0, 255, 0) if mask is None or mask[idx] else (0, 0, 255)
    #         pt1 = (int(x1), int(y1))
    #         pt2 = (int(x2) + w1, int(y2))
    #         cv2.circle(vis, pt1, 3, color, -1)
    #         cv2.circle(vis, pt2, 3, color, -1)
    #         cv2.line(vis, pt1, pt2, color, 1)
    #
    #     return vis

# Implementation
    #  Detect corners for ALL images first

    # for img in enumerate(images):
    #     all_corners, all_responses = detect_corners(img, num_features)
    #
    #
    #
    # # 3. Apply ANMS to all detected corners
    # print("\n2. Applying ANMS to all images")
    # all_anms_corners = []
    # for i, (corners, response) in enumerate(zip(all_corners, all_responses)):
    #     print(f"Processing image {i + 1}/{len(images)}")
    #     anms_corners = adaptive_non_maximal_suppression(response, corners, Args.NumFeatures)
    #
    #     # Save ANMS visualization
    #     vis = images[i].copy()
    #     for x, y in anms_corners.astype(np.int32):
    #         cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    #     cv2.imwrite(f'{Args.output_dir}/anms_{i + 1}.png', vis)
    #
    #     all_anms_corners.append(anms_corners)
    #
    # # 4. Extract features for ALL images
    # print("\n3. Extracting features for all images")
    # all_descriptors = []
    # all_valid_corners = []
    # for i, (img, corners) in enumerate(zip(images, all_anms_corners)):
    #     print(f"Processing image {i + 1}/{len(images)}")
    #     descriptors, valid_corners = extract_features(img, corners)
    #
    #     # Save feature extraction visualization
    #     vis = img.copy()
    #     for x, y in valid_corners.astype(np.int32):
    #         cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    #         cv2.rectangle(vis, (int(x) - 20, int(y) - 20),
    #                       (int(x) + 20, int(y) + 20), (0, 255, 0), 1)
    #     cv2.imwrite(f'{Args.output_dir}/features_{i + 1}.png', vis)
    #
    #     all_descriptors.append(descriptors)
    #     all_valid_corners.append(valid_corners)
    #
    # # Initialize panorama with first image
    # panorama = images[0]
    #
    # # Process image pairs
    # for i in range(1, len(images)):
    #     print(f"\nProcessing image pair {i}/{len(images) - 1}")
    #
    #     # 5. Match features between consecutive images
    #     print("4. Feature Matching")
    #     matches = match_features(
    #         all_descriptors[i - 1], all_descriptors[i],
    #         all_valid_corners[i - 1], all_valid_corners[i]
    #     )
    #
    #     if len(matches) < 4:
    #         print(f"Too few matches ({len(matches)}) for image pair {i}")
    #         continue
    #
    #     # Save matching visualization
    #     match_vis = create_matching_visualization(images[i - 1], images[i], matches)
    #     cv2.imwrite(f'{Args.output_dir}/matches_{i}.png', match_vis)
    #
    #     # 6. RANSAC using matched features
    #     print("5. RANSAC Homography Estimation")
    #     H, mask = ransac_homography(matches)
    #
    #
    #     if H is None:
    #         print(f"Failed to find homography for image pair {i}")
    #         continue
    #
    #     # Save RANSAC visualization
    #     ransac_vis = create_matching_visualization(images[i - 1], images[i], matches, mask)
    #     cv2.imwrite(f'{Args.output_dir}/ransac_{i}.png', ransac_vis)
    #
    #     # 7. Image stitching using RANSAC output
    #     print("6. Image Blending")
    #     try:
    #         panorama = blend_images(panorama, images[i], H, Args.output_dir, i)
    #         cv2.imwrite(f'{Args.output_dir}/panorama_step_{i}.png', panorama)
    #     except Exception as e:
    #         print(f"Error in blending: {str(e)}")
    #         continue
    #
    # # Save final panorama
    # cv2.imwrite(f'{Args.output_dir}/final_panorama.png', panorama)
    # print("\nPanorama creation completed!")

    # Create panorama
    # Create panorama
    panorama = create_panorama(images, Args.NumFeatures)

    # Save result
    cv2.imwrite(os.path.join(Args.output_dir, 'panorama3.jpg'), panorama)

    # # Initialize panorama
    # panorama = images[0]
    #
    # # Process image pairs
    # for i in range(1, len(images)):
    #     matches = match_features(all_descriptors[i - 1], all_descriptors[i], valid_corners[i - 1], valid_corners[i])
    #
    #     if len(matches) < 10:
    #         print(f"Too few matches for image {i}")
    #         continue
    #
    #     # Save matching visualization
    #     vis = create_matching_visualization(matches, np.ones(len(matches)), i)
    #     cv2.imwrite(f'{Args.output_dir}/matches_{i}.png', vis)
    #
    #     # RANSAC
    #     H, mask = ransac_homography(
    #         matches, output_dir=Args.output_dir, img_idx=i
    #     )
    #
    #     if H is None:
    #         print(f"Failed to find homography for image {i}")
    #         continue
    #
    #     # Blend images
    #     try:
    #         panorama = blend_images(panorama, images[i], H)
    #         cv2.imwrite(f'{Args.output_dir}/panorama_{i}.png', panorama)
    #     except Exception as e:
    #         print(f"Error blending image {i}: {str(e)}")
    #         continue
    #
    # cv2.imwrite(f'{Args.output_dir}/final_panorama.png', panorama)
    # print("Panorama creation completed")


if __name__ == "__main__":
    main()
