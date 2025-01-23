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
    Parser.add_argument('--folder', default='../Data/Train/Set1', help='Path to image folder')
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

    def detect_corners(img, num_features):

        print("\n1. Corner Detection for all images")
        all_corners = []
        all_responses = []
        for i, img in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            # Detect corners using goodFeaturesToTrack
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=num_features,
                qualityLevel=0.01,
                minDistance=10
            )
            corners = corners.reshape(-1, 2)
            # Get corner response for ANMS
            response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

            # Save corner detection visualization
            vis = img.copy()
            for x, y in corners.astype(np.int32):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imwrite(f'{Args.output_dir}/corners_{i + 1}.png', vis)

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

        # Initialize r_i = infinity for i = [1:N_strong]
        r = np.full(N_strong, np.inf)

        # Get corner scores
        scores = np.array([C_img[int(y), int(x)] for x, y in corners])

        # For each corner
        for i in range(N_strong):
            for j in range(N_strong):
                if scores[j] > scores[i]:
                    # Compute Euclidean distance
                    ED = (corners[j, 0] - corners[i, 0]) ** 2 + \
                         (corners[j, 1] - corners[i, 1]) ** 2
                    r[i] = min(r[i], ED)

        # Sort r_i in descending order and pick top N_best points
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

        for x, y in corners:
            x, y = int(x), int(y)

            # Check if patch fits within image
            if (y - 20 < 0 or y + 21 > img.shape[0] or
                    x - 20 < 0 or x + 21 > img.shape[1]):
                continue

            # Extract 41x41 patch
            patch = img[y - 20:y + 21, x - 20:x + 21]

            # Apply Gaussian blur
            patch = cv2.GaussianBlur(patch, (3, 3), 0)

            # Resize to 8x8
            patch = cv2.resize(patch, (8, 8))

            # Convert to feature vector
            feat = patch.reshape(-1)
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

        matches = []

        for i, desc in enumerate(desc1):
            # Compute distances to all descriptors in desc2
            distances = np.sum((desc2 - desc) ** 2, axis=1)
            idx = np.argsort(distances)

            # Apply ratio test
            if distances[idx[0]] < 0.8 * distances[idx[1]]:
                matches.append((corners1[i], corners2[idx[0]]))

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

    def ransac_homography(matches, threshold=5.0, max_iters=1000):

        if len(matches) < 4:
            return None, None

        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        best_H = None
        best_mask = None
        best_inliers = 0

        for _ in range(max_iters):
            # 1. Select random points
            idx = np.random.choice(len(matches), 4, replace=False)
            src_pts = pts1[idx].reshape(-1, 1, 2).astype(np.float32)
            dst_pts = pts2[idx].reshape(-1, 1, 2).astype(np.float32)

            # Compute homography
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            if H is None:
                continue

            # Transform points and compute SSD
            transformed = cv2.perspectiveTransform(
                pts1.reshape(-1, 1, 2).astype(np.float32), H
            )

            if transformed is None:
                continue

            ssd = np.sum((pts2.reshape(-1, 1, 2) - transformed) ** 2, axis=(1, 2))
            mask = ssd < threshold
            inliers = np.sum(mask)

            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
                best_mask = mask

            # Early termination if 90% inliers found
            if best_inliers > len(matches) * 0.9:
                break

            # Ensure homography matrix is float32
            if best_H is not None:
                best_H = best_H.astype(np.float32)

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

    def blend_images(img1, img2, H, output_dir, idx):
        """
        Enhanced blending with Gaussian feathering and Poisson blending
        """
        # Convert H to float32
        H = H.astype(np.float32)

        # Calculate output bounds
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        all_corners = np.concatenate((
            corners2,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        ))

        xmin, ymin = np.int32(np.min(all_corners, axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(np.max(all_corners, axis=0).ravel() + 0.5)

        # Create translation matrix
        translation = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ], dtype=np.float32)

        # Final transformation
        H_final = translation @ H
        output_shape = (xmax - xmin, ymax - ymin)

        # Warp images
        warped1 = cv2.warpPerspective(img1, H_final, output_shape)
        warped2 = cv2.warpPerspective(img2, translation, output_shape)

        # Create masks
        mask1 = cv2.warpPerspective(
            np.ones_like(img1[:, :, 0], dtype=np.float32),
            H_final, output_shape
        )
        mask2 = cv2.warpPerspective(
            np.ones_like(img2[:, :, 0], dtype=np.float32),
            translation, output_shape
        )

        # Feather masks
        kernel_size = 51
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size // 4)
        kernel_2d = kernel @ kernel.T

        mask1_feather = cv2.filter2D(mask1, -1, kernel_2d)
        mask2_feather = cv2.filter2D(mask2, -1, kernel_2d)

        # Normalize masks
        sum_masks = mask1_feather + mask2_feather
        mask1_norm = np.divide(mask1_feather, sum_masks,
                               out=np.zeros_like(mask1_feather), where=sum_masks != 0)
        mask2_norm = np.divide(mask2_feather, sum_masks,
                               out=np.zeros_like(mask2_feather), where=sum_masks != 0)

        # Initial blend
        result = np.zeros_like(warped1, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = (warped1[:, :, c] * mask1_norm +
                               warped2[:, :, c] * mask2_norm)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Poisson blending in overlap region
        overlap = (mask1 > 0) & (mask2 > 0)
        if np.sum(overlap) > 0:
            overlap_mask = overlap.astype(np.uint8) * 255
            center = (output_shape[0] // 2, output_shape[1] // 2)
            try:
                result = cv2.seamlessClone(warped2, result, overlap_mask, center, cv2.MIXED_CLONE)
            except cv2.error:
                print("Warning: Poisson blending failed")

        # Save intermediate results
        cv2.imwrite(f'{output_dir}/warped1_{idx}.png', warped1)
        cv2.imwrite(f'{output_dir}/warped2_{idx}.png', warped2)
        cv2.imwrite(f'{output_dir}/blended_{idx}.png', result)

        return result

    def create_panorama(images, num_features, output_dir):
        """
        Create panorama from multiple images using existing feature detection pipeline
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        # 1. Detect corners for all images
        all_corners, all_responses = detect_corners(images[0], num_features)

        # 2. Apply ANMS
        all_anms_corners = []
        for i, (corners, response) in enumerate(zip(all_corners, all_responses)):
            anms_corners = adaptive_non_maximal_suppression(response, corners, num_features)
            all_anms_corners.append(anms_corners)

        # 3. Extract features
        all_descriptors = []
        all_valid_corners = []
        for i, (img, corners) in enumerate(zip(images, all_anms_corners)):
            descriptors, valid_corners = extract_features(img, corners)
            all_descriptors.append(descriptors)
            all_valid_corners.append(valid_corners)

        # Initialize panorama
        panorama = images[0]
        valid_images = [0]

        # Process image pairs
        for i in range(1, len(images)):
            # Match features
            matches = match_features(
                all_descriptors[i - 1], all_descriptors[i],
                all_valid_corners[i - 1], all_valid_corners[i]
            )

            if len(matches) < 10:
                print(f"Insufficient matches for image {i}")
                continue

            # RANSAC
            H, mask = ransac_homography(matches)
            if H is None or np.abs(np.linalg.det(H) - 1) > 0.1:
                print(f"Invalid homography for image {i}")
                continue

            # Blend images
            try:
                panorama = blend_images(panorama, images[i], H, output_dir, i)
                valid_images.append(i)
            except Exception as e:
                print(f"Blending error for image {i}: {str(e)}")
                continue

        if len(valid_images) < 2:
            raise RuntimeError("Failed to create panorama")

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

    def create_matching_visualization(img1, img2, matches, mask=None):

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2

        for idx, ((x1, y1), (x2, y2)) in enumerate(matches):
            color = (0, 255, 0) if mask is None or mask[idx] else (0, 0, 255)
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2) + w1, int(y2))
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)

        return vis

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
    panorama = create_panorama(images, int(Args.NumFeatures), Args.output_dir)
    cv2.imwrite(f'{Args.output_dir}/final_panorama.png', panorama)

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
