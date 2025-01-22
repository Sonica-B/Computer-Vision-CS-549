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


    num_features = int(Args.NumFeatures)
    os.makedirs(Args.output_dir, exist_ok=True)

    """
    Read a set of images for Panorama stitching
    """

    # Read images
    images = [cv2.imread(os.path.join(Args.folder, f)) for f in sorted(os.listdir(Args.folder)) if f.endswith('.jpg')]
    if len(images) < 3:
        print("Need at least 3 images for panorama creation.")
        return

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    def detect_corners_all_images(images, num_features, output_dir='Outputs'):
        print("Step 1: Detecting corners in all images...")
        all_corners = []
        all_responses = []

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=num_features,
                qualityLevel=0.01,
                minDistance=10
            )

            if corners is None:
                print(f"No corners detected in image {i + 1}")
                continue

            # Compute corner responses
            response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            corners = corners.reshape(-1, 2)

            # Save visualization
            vis = img.copy()
            for x, y in corners.astype(np.int32):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imwrite(f'{output_dir}/corners_{i + 1}.png', vis)
            print(f"Corners detected and saved for image {i + 1}.")

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

    def adaptive_non_maximal_suppression(C_img, corners, num_best):
        N_strong = len(corners)
        if N_strong <= num_best:
            return corners

        r = np.full(N_strong, np.inf)
        scores = np.array([C_img[int(y), int(x)] for x, y in corners])

        for i in range(N_strong):
            for j in range(N_strong):
                if scores[j] > scores[i]:
                    ED = (corners[j, 0] - corners[i, 0]) ** 2 + (corners[j, 1] - corners[i, 1]) ** 2
                    r[i] = min(r[i], ED)

        idx = np.argsort(r)[::-1][:num_best]
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

    def extract_features(img, corners, output_dir, img_idx):
        print(f"Step 3: Extracting features for image {img_idx + 1}...")
        descriptors = []
        vis = img.copy()
        for x, y in corners:
            x, y = int(x), int(y)

            patch_x1 = max(x - 20, 0)
            patch_x2 = min(x + 21, img.shape[1])
            patch_y1 = max(y - 20, 0)
            patch_y2 = min(y + 21, img.shape[0])
            patch = img[patch_y1:patch_y2, patch_x1:patch_x2]

            patch = cv2.GaussianBlur(patch, (3, 3), 0)
            patch = cv2.resize(patch, (8, 8))

            feat = patch.flatten()
            feat = (feat - feat.mean()) / (feat.std() + 1e-7)
            descriptors.append(feat)

            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
            cv2.rectangle(vis, (x - 20, y - 20), (x + 20, y + 20), (255, 0, 0), 1)

        cv2.imwrite(f'{output_dir}/features_{img_idx + 1}.png', vis)
        print(f"Features extracted and visualization saved for image {img_idx + 1}.")

        return np.array(descriptors)
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


    def match_features(desc1, desc2, corners1, corners2, output_dir, img_idx):
        print(f"Step 4: Matching features between images {img_idx} and {img_idx + 1}...")
        matches = []
        for i, desc in enumerate(desc1):
            dists = np.sum((desc2 - desc) ** 2, axis=1)

            idx = np.argsort(dists)
            if dists[idx[0]] < 0.8 * dists[idx[1]]:
                matches.append((corners1[i], corners2[idx[0]]))

        vis = np.zeros((500, 500, 3), dtype=np.uint8)
        for pt1, pt2 in matches:
            cv2.circle(vis, (int(pt1[0]), int(pt1[1])), 3, (0, 255, 0), -1)
            cv2.circle(vis, (int(pt2[0]), int(pt2[1])), 3, (0, 0, 255), -1)
        cv2.imwrite(f'{output_dir}/matches_{img_idx + 1}.png', vis)
        print(f"Feature matching completed and visualization saved for image {img_idx}.")

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

    def ransac_homography(matches, threshold=5.0, max_iters=1000):
        print("Step 5: Running RANSAC to estimate homography...")
        if len(matches) < 4:
            print("Not enough matches for RANSAC.")
            return None, None

        pts1 = np.array([m[0] for m in matches], dtype=np.float32)
        pts2 = np.array([m[1] for m in matches], dtype=np.float32)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)

        if H is None or mask is None:
            print("RANSAC failed to find a valid homography.")
            return None, None

        inlier_count = np.sum(mask)
        print(f"RANSAC succeeded with {inlier_count} inliers.")

        return H, mask.flatten()

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

    def blend_images(img1, img2, H, output_dir, img_idx):
        print(f"Step 6: Blending image {img_idx + 1} with the panorama...")
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners2 = corners2.reshape(-1, 2)  # Flatten to match 2D corner dimensions
        corners = np.vstack((corners2, [[0, 0], [w2, 0], [w2, h2], [0, h2]]))

        [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel())
        translation = np.float32([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

        warped1 = cv2.warpPerspective(img1, translation @ H, (xmax - xmin, ymax - ymin))
        warped2 = cv2.warpPerspective(img2, translation, (xmax - xmin, ymax - ymin))

        result = np.where(warped2 > 0, warped2, warped1)

        cv2.imwrite(f'{output_dir}/panorama_{img_idx + 1}.png', result)
        print(f"Panorama blended and saved for image {img_idx + 1}.")

        return result

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


# Implementation

    all_corners, all_responses = detect_corners_all_images(images, num_features, Args.output_dir)

    all_anms_corners = []
    for i, (corners, response) in enumerate(zip(all_corners, all_responses)):
        anms_corners = adaptive_non_maximal_suppression(response, corners, num_features)
        all_anms_corners.append(anms_corners)

    all_descriptors = []
    for i, (img, corners) in enumerate(zip(images, all_anms_corners)):
        desc = extract_features(img, corners, Args.output_dir, i)
        all_descriptors.append(desc)

    panorama = images[0]
    for i in range(1, len(images)):
        matches = match_features(all_descriptors[i - 1], all_descriptors[i], all_anms_corners[i - 1], all_anms_corners[i], Args.output_dir, i)

        if len(matches) < 4:
            print(f"Too few matches for image {i}, skipping.")
            continue
        print(f"Number of matches: {len(matches)}")
        print(f"Matches: {matches[:5]}")

        H, mask = ransac_homography(matches)
        if H is None:
            print(f"Skipping image {i} due to invalid homography.")
            continue

        panorama = blend_images(panorama, images[i], H, Args.output_dir, i)

    cv2.imwrite(f'{Args.output_dir}/final_panorama.png', panorama)
    print("Panorama creation completed.")

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
