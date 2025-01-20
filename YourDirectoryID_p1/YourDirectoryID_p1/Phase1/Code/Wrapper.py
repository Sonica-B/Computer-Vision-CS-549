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
    NumFeatures = int(Args.NumFeatures)
    folder_path = Args.folder

    # Check for CUDA availability
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA device found! Using GPU acceleration")
        use_gpu = True
    else:
        print("No CUDA device found. Using CPU")
        use_gpu = False

    """
    Read a set of images for Panorama stitching
    """

    print("\nReading images...")
    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    for f in image_files:
        print(f"Reading {f}")
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            if use_gpu:
                # Upload to GPU memory
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                images.append(gpu_img)
            else:
                images.append(img)

    if len(images) < 3:
        print("Need at least 3 images")
        return

    print(f"\nProcessing first image...")

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    def detect_corners(img):
        """
        Detect corners using Harris corners
        Args:
            img: Input image
        Returns:
            C_img: Corner response image from cornermetric
            corners: Corner coordinates
            scores: Corner response scores
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # Compute Harris corner response image
        C_img = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Normalize corner response
        C_img_norm = cv2.normalize(C_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Find local maxima
        corners = []
        scores = []

        # Pad the corner response image to handle boundaries
        padded = np.pad(C_img_norm, ((1, 1), (1, 1)), mode='constant')

        # Find local maxima
        for i in range(1, C_img_norm.shape[0] + 1):
            for j in range(1, C_img_norm.shape[1] + 1):
                # 3x3 window for local maximum check
                window = padded[i - 1:i + 2, j - 1:j + 2]
                center = window[1, 1]

                # Check if center is local maximum and above threshold
                if center > 0.01 * C_img_norm.max() and center == np.max(window):
                    # Note: x=j-1, y=i-1 to convert back to image coordinates
                    corners.append([j - 1, i - 1])  # Store as [x,y]
                    scores.append(center)

        return C_img_norm, np.array(corners), np.array(scores)

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
        # Find all local maxima
        h, w = C_img.shape
        local_maxima = []
        scores = []

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Get 3x3 window
                window = C_img[i - 1:i + 2, j - 1:j + 2]
                center = C_img[i, j]

                # Check if local maximum
                if center == np.max(window) and center > 0:
                    # Store [j,i] as [x,y] coordinates
                    local_maxima.append([j, i])  # x=j, y=i
                    scores.append(center)

        local_maxima = np.array(local_maxima)
        N_strong = len(local_maxima)

        if N_strong < 4:  # Need at least 4 corners for homography
            return local_maxima

        # Initialize r_i = infinity for i = [1:N_strong]
        r = np.full(N_strong, np.inf)

        # For each corner
        for i in range(N_strong):
            x_i, y_i = local_maxima[i]

            # Compare with all other corners
            for j in range(N_strong):
                if i == j:  # Skip self-comparison
                    continue

                x_j, y_j = local_maxima[j]

                # Compare corner responses
                if C_img[y_j, x_j] > C_img[y_i, x_i]:
                    # Calculate Euclidean distance
                    ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
                    # Update r_i if necessary
                    r[i] = min(r[i], ED)

        # Sort r_i in descending order and pick top N_best points
        idx = np.argsort(r)[::-1]
        idx = idx[:min(N_best, N_strong)]

        return local_maxima[idx]

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    def extract_features(img, corners):
        """
        Extract feature descriptors
        Args:
            img: Input image
            corners: Corner points
        Returns:
            descriptors: Feature descriptors
            valid_corners: Valid corner points
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        descriptors = []
        valid_corners = []

        for x, y in corners:
            # Check bounds for 41x41 patch
            x, y = int(x), int(y)
            if y - 20 < 0 or y + 21 > gray.shape[0] or x - 20 < 0 or x + 21 > gray.shape[1]:
                continue

            # Extract 41x41 patch
            patch = gray[y - 20:y + 21, x - 20:x + 21]

            # Apply Gaussian blur
            patch = cv2.GaussianBlur(patch, (3, 3), 0)

            # Subsample to 8x8
            patch = cv2.resize(patch, (8, 8))

            # Normalize descriptor
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
            # Compute SSD with all descriptors in desc2
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
        Estimate homography using RANSAC following the steps:
        1. Select four feature pairs (at random)
        2. Compute homography H between these pairs
        3. Compute inliers using SSD < τ
        4. Repeat until max_iters or >90% inliers found
        5. Keep largest set of inliers
        6. Re-compute least-squares H estimate on all inliers
        """
        if len(matches) < 4:
            return None, None

        p = np.float32(matches[:, 0])  # p_i from image 1
        p_prime = np.float32(matches[:, 1])  # p'_i from image 2

        best_H = None
        best_mask = None
        best_inliers = 0

        for _ in range(max_iters):
            # 1. Select four feature pairs
            idx = np.random.choice(len(matches), 4, replace=False)
            src_pts = p[idx].reshape(-1, 1, 2)
            dst_pts = p_prime[idx].reshape(-1, 1, 2)

            # 2. Compute homography
            H = cv2.findHomography(src_pts, dst_pts)[0]
            if H is None:
                continue

            # 3. Compute SSD for all points
            transformed = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H)
            if transformed is None:
                continue

            # SSD calculation
            ssd = np.sum((p_prime.reshape(-1, 1, 2) - transformed) ** 2, axis=(1, 2))
            mask = ssd < threshold
            inliers = np.sum(mask)

            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
                best_mask = mask

            # Check for 90% inliers
            if best_inliers > len(p) * 0.9:
                break

        # 5-6. Use largest inlier set to recompute H
        if best_mask is not None and np.sum(best_mask) >= 4:
            src_pts = p[best_mask].reshape(-1, 1, 2)
            dst_pts = p_prime[best_mask].reshape(-1, 1, 2)
            best_H = cv2.findHomography(src_pts, dst_pts)[0]

        return best_H, best_mask

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    def blend_images(img1, img2, H):
        """
        Blend images using homography
        """
        # Convert H to float32
        H = H.astype(np.float32)

        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Transform corners
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)))

        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

        # Translation matrix
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

        # Warp images
        warped1 = cv2.warpPerspective(img1, Ht @ H, (xmax - xmin, ymax - ymin))
        warped2 = cv2.warpPerspective(img2, Ht, (xmax - xmin, ymax - ymin))

        # Create masks
        mask1 = cv2.warpPerspective(np.ones_like(img1[:, :, 0]), Ht @ H, (xmax - xmin, ymax - ymin))
        mask2 = cv2.warpPerspective(np.ones_like(img2[:, :, 0]), Ht, (xmax - xmin, ymax - ymin))

        # Blend images
        result = np.zeros_like(warped1, dtype=np.float32)

        # Non-overlapping regions
        result[mask1 > 0] = warped1[mask1 > 0]
        result[mask2 > 0] = warped2[mask2 > 0]

        # Overlapping regions
        overlap = (mask1 > 0) & (mask2 > 0)
        if overlap.any():
            alpha = 0.5
            result[overlap] = (alpha * warped1[overlap] + (1 - alpha) * warped2[overlap])

        return result.astype(np.uint8)


# Implementation
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    print("1. Detecting corners...")
    if use_gpu:
        img0 = images[0].download()
    else:
        img0 = images[0]
    C_img, corners, scores = detect_corners(img0)

    print(f"Found {len(corners) if corners is not None else 0} corners")

    # Save corner detection visualization
    corner_vis = img0.copy()
    if corners is not None and len(corners) > 0:
        for x, y in corners.astype(np.int32):
            cv2.circle(corner_vis, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite('corners.png', corner_vis)
    print("Saved corners.png")

    print("\n2. Performing ANMS...")
    anms_corners = adaptiveNonMaximalSuppression(C_img, NumFeatures)
    print(f"Selected {len(anms_corners)} corners after ANMS")

    anms_vis = img0.copy()
    if len(anms_corners) > 0:
        for x, y in anms_corners.astype(np.int32):
            cv2.circle(anms_vis, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite('anms.png', anms_vis)
    print("Saved anms.png")

    print("\n3. Extracting feature descriptors...")
    desc1, valid_corners1 = extract_features(img0, anms_corners)
    print(f"Extracted {len(desc1)} valid features")

    fd_vis = img0.copy()
    if len(valid_corners1) > 0:
        for x, y in valid_corners1.astype(np.int32):
            cv2.circle(fd_vis, (x, y), 3, (0, 255, 0), -1)
            cv2.rectangle(fd_vis, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 1)
    cv2.imwrite('FD.png', fd_vis)
    print("Saved FD.png")

    # Initialize panorama with first image
    panorama = img0

    # For each subsequent image
    print("\nProcessing remaining images...")
    for i in range(1, len(images)):
        print(f"\nProcessing image pair {i}/{len(images) - 1}")

        print("1. Getting features from current panorama...")
        C_img1, corners1, scores1 = detect_corners(panorama)
        anms_corners1 = adaptiveNonMaximalSuppression(C_img1, NumFeatures)
        desc1, valid_corners1 = extract_features(panorama, anms_corners1)

        print("2. Getting features from next image...")
        if use_gpu:
            img_next = images[i].download()
        else:
            img_next = images[i]
        C_img2, corners2, scores2 = detect_corners(img_next)
        anms_corners2 = adaptiveNonMaximalSuppression(C_img2, NumFeatures)
        desc2, valid_corners2 = extract_features(img_next, anms_corners2)

        print("3. Matching features...")
        matches = match_features(desc1, desc2, valid_corners1, valid_corners2)
        print(f"Found {len(matches)} matches")

        if len(matches) > 0:
            # Create match visualization
            h_max = max(panorama.shape[0], img_next.shape[0])
            w_total = panorama.shape[1] + img_next.shape[1]
            match_vis = np.zeros((h_max, w_total, 3), dtype=np.uint8)

            # Copy images
            match_vis[:panorama.shape[0], :panorama.shape[1]] = panorama
            match_vis[:img_next.shape[0], panorama.shape[1]:] = img_next

            # Draw matches
            for (x1, y1), (x2, y2) in matches:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2 + panorama.shape[1]), int(y2))
                cv2.line(match_vis, pt1, pt2, (0, 255, 0), 1)

            cv2.imwrite(f'matching_{i}.png', match_vis)
            print(f"Saved matching_{i}.png")

            print("4. Computing homography using RANSAC...")
            H, mask = ransac_homography(matches)

            if H is not None:
                print("5. Blending images...")
                try:
                    panorama = blend_images(panorama, img_next, H)
                    cv2.imwrite(f'panorama_{i}.png', panorama)
                    print(f"Saved panorama_{i}.png")
                except Exception as e:
                    print(f"Error blending images: {str(e)}")
                    continue
            else:
                print(f"Failed to compute homography")
        else:
            print("No matches found between images")

    # Save final panorama
    cv2.imwrite('mypano.png', panorama)
    print("\nPanorama creation completed! Final result saved as mypano.png")

if __name__ == "__main__":
    main()
