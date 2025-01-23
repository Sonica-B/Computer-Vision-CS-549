import numpy as np
import cv2
import argparse
import os
import torch


def detect_corners(image, num_features):
    """Corner detection with Harris corners"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Get corner response using Harris
    response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Find local maxima
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=num_features,
        qualityLevel=0.01,
        minDistance=10
    )
    corners = corners.reshape(-1, 2)

    return corners, response


def adaptive_non_maximal_suppression(C_img, corners, N_best):
    """
    ANMS algorithm exactly as specified in Fig 3 of project PDF
    """
    # Initialize r_i = infinity for i = [1:N_strong]
    N_strong = len(corners)
    if N_strong <= N_best:
        return corners

    r = np.full(N_strong, np.inf)

    # Get corner scores
    scores = np.array([C_img[int(y), int(x)] for x, y in corners])

    # Compare each corner against all other corners
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


def extract_features(img, corners):
    """Extract feature descriptors for each corner"""
    descriptors = []
    valid_corners = []

    for x, y in corners:
        x, y = int(x), int(y)

        # Check patch bounds
        if (y - 20 < 0 or y + 21 > img.shape[0] or
                x - 20 < 0 or x + 21 > img.shape[1]):
            continue

        # Extract and process 41x41 patch
        patch = img[y - 20:y + 21, x - 20:x + 21]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        patch = cv2.resize(patch, (8, 8))

        # Create feature vector
        feat = patch.reshape(-1)
        feat = (feat - feat.mean()) / (feat.std() + 1e-7)

        descriptors.append(feat)
        valid_corners.append([x, y])

    return np.array(descriptors), np.array(valid_corners)


def match_features(desc1, desc2, corners1, corners2):
    """Match features using ratio test"""
    matches = []

    for i, desc in enumerate(desc1):
        # Compute L2 distances to all descriptors in desc2
        distances = np.sum((desc2 - desc) ** 2, axis=1)
        idx = np.argsort(distances)

        # Apply ratio test
        if distances[idx[0]] < 0.8 * distances[idx[1]]:
            matches.append((corners1[i], corners2[idx[0]]))

    return np.array(matches)


def ransac_homography(matches, threshold=5.0, max_iters=1000):
    """RANSAC for robust homography estimation"""
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
        src = pts1[idx].reshape(-1, 1, 2).astype(np.float32)
        dst = pts2[idx].reshape(-1, 1, 2).astype(np.float32)

        # 2. Compute homography
        H = cv2.getPerspectiveTransform(src, dst)
        if H is None:
            continue

        # 3. Transform points
        transformed = cv2.perspectiveTransform(
            pts1.reshape(-1, 1, 2).astype(np.float32),
            H
        )
        if transformed is None:
            continue

        # 4. Count inliers
        errors = np.sqrt(np.sum((pts2.reshape(-1, 1, 2) - transformed) ** 2, axis=(1, 2)))
        mask = errors < threshold
        inliers = np.sum(mask)

        # 5. Update best model
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H.astype(np.float32)
            best_mask = mask

        # Early stopping if we found a very good model
        if best_inliers > len(matches) * 0.9:
            break

    return best_H, best_mask


def blend_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of first image
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = cv2.perspectiveTransform(corners1, H)

    # Stack corners with consistent dimensions
    corners2 = corners2.reshape(-1, 2)
    img2_corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    all_corners = np.vstack((corners2, img2_corners))

    xmin, ymin = np.int32(np.min(all_corners, axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(np.max(all_corners, axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float32)

    output_shape = (xmax - xmin, ymax - ymin)
    warped1 = cv2.warpPerspective(img1, translation @ H, output_shape)
    warped2 = cv2.warpPerspective(img2, translation, output_shape)

    # Create masks
    mask1 = cv2.warpPerspective(
        np.ones_like(img1[:, :, 0], dtype=np.float32),
        translation @ H, output_shape
    )
    mask2 = cv2.warpPerspective(
        np.ones_like(img2[:, :, 0], dtype=np.float32),
        translation, output_shape
    )

    # Gaussian feathering
    kernel_size = 51
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size // 4)
    kernel_2d = kernel @ kernel.T

    mask1_feather = cv2.filter2D(mask1, -1, kernel_2d)
    mask2_feather = cv2.filter2D(mask2, -1, kernel_2d)

    # Normalize masks
    sum_masks = mask1_feather + mask2_feather
    mask1_norm = np.divide(mask1_feather, sum_masks,
                           out=np.zeros_like(mask1_feather),
                           where=sum_masks != 0)
    mask2_norm = np.divide(mask2_feather, sum_masks,
                           out=np.zeros_like(mask2_feather),
                           where=sum_masks != 0)

    # Initial blend
    result = np.zeros_like(warped1, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = warped1[:, :, c] * mask1_norm + warped2[:, :, c] * mask2_norm
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Poisson blending
    overlap = (mask1 > 0) & (mask2 > 0)
    if np.sum(overlap) > 1000:
        overlap_mask = overlap.astype(np.uint8) * 255
        y, x = np.nonzero(overlap)
        if len(x) > 0 and len(y) > 0:
            center = (int(np.mean(x)), int(np.mean(y)))
            try:
                result = cv2.seamlessClone(
                    warped2, result, overlap_mask,
                    center, cv2.MIXED_CLONE
                )
            except cv2.error:
                print("Warning: Poisson blending failed")

    return result


def create_panorama(images, num_features=1000):
    """
    Create panorama from multiple images
    """
    if len(images) < 2:
        return images[0]

    # Initialize panorama with first image
    panorama = images[0]

    # Process each additional image
    for i in range(1, len(images)):
        # Detect and match features
        corners1, response1 = detect_corners(panorama, num_features)
        corners2, response2 = detect_corners(images[i], num_features)

        corners1 = adaptive_non_maximal_suppression(response1, corners1, num_features)
        corners2 = adaptive_non_maximal_suppression(response2, corners2, num_features)

        desc1, valid_corners1 = extract_features(panorama, corners1)
        desc2, valid_corners2 = extract_features(images[i], corners2)

        matches = match_features(desc1, desc2, valid_corners1, valid_corners2)

        # Skip if too few matches
        if len(matches) < 10:
            print(f"Skipping image {i} - insufficient matches ({len(matches)})")
            continue

        # Estimate homography
        H, mask = ransac_homography(matches)
        if H is None:
            print(f"Skipping image {i} - failed to estimate homography")
            continue

        # Blend images
        panorama = blend_images(panorama, images[i], H)

    return panorama


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumFeatures', default=1000, type=int,
                        help='Number of features to detect')
    parser.add_argument('--folder', default='../Data/Train/Set1',
                        help='Path to image folder')
    parser.add_argument('--output_dir', default='Outputs',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read images
    images = []
    for f in sorted(os.listdir(args.folder)):
        if f.endswith('.jpg'):
            img = cv2.imread(os.path.join(args.folder, f))
            if img is not None:
                images.append(img)

    if len(images) < 2:
        print("Need at least 2 images")
        return

    # Create panorama
    panorama = create_panorama(images, args.NumFeatures)

    # Save result
    cv2.imwrite(os.path.join(args.output_dir, 'panorama.jpg'), panorama)


if __name__ == '__main__':
    main()