import numpy as np
import cv2
import argparse
import os
import torch


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

    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float32)

    output_shape = (xmax - xmin, ymax - ymin)
    H_final = translation @ H

    # Convert to LAB color space for better color blending
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    warped1 = cv2.warpPerspective(img1_lab, H_final, output_shape, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    warped2 = cv2.warpPerspective(img2_lab, translation, output_shape, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    mask1 = cv2.warpPerspective(np.ones_like(img1[:, :, 0]), H_final, output_shape)
    mask2 = cv2.warpPerspective(np.ones_like(img2[:, :, 0]), translation, output_shape)

    # Multi-band blending
    num_levels = 5
    gaussian1 = [warped1.astype(np.float32)]
    gaussian2 = [warped2.astype(np.float32)]
    mask1_pyr = [mask1.astype(np.float32)]
    mask2_pyr = [mask2.astype(np.float32)]

    for i in range(num_levels):
        gaussian1.append(cv2.pyrDown(gaussian1[-1]))
        gaussian2.append(cv2.pyrDown(gaussian2[-1]))
        mask1_pyr.append(cv2.pyrDown(mask1_pyr[-1]))
        mask2_pyr.append(cv2.pyrDown(mask2_pyr[-1]))

    laplacian1 = [gaussian1[-1]]
    laplacian2 = [gaussian2[-1]]

    for i in range(num_levels - 1, 0, -1):
        size = (gaussian1[i - 1].shape[1], gaussian1[i - 1].shape[0])
        expanded = cv2.pyrUp(gaussian1[i], dstsize=size)
        laplacian = cv2.subtract(gaussian1[i - 1], expanded)
        laplacian1.append(laplacian)

        expanded = cv2.pyrUp(gaussian2[i], dstsize=size)
        laplacian = cv2.subtract(gaussian2[i - 1], expanded)
        laplacian2.append(laplacian)

    # Blend pyramids
    blended_pyr = []
    kernel = cv2.getGaussianKernel(51, 12)
    kernel_2d = kernel @ kernel.T

    for i in range(num_levels):
        mask1_feather = cv2.filter2D(mask1_pyr[num_levels - 1 - i], -1, kernel_2d)
        mask2_feather = cv2.filter2D(mask2_pyr[num_levels - 1 - i], -1, kernel_2d)

        mask_sum = mask1_feather + mask2_feather
        weight1 = mask1_feather / (mask_sum + 1e-10)
        weight2 = mask2_feather / (mask_sum + 1e-10)

        blended = laplacian1[i] * weight1[..., np.newaxis] + laplacian2[i] * weight2[..., np.newaxis]
        blended_pyr.append(blended)

    # Reconstruct final image
    result_lab = blended_pyr[0]
    for i in range(1, num_levels):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        result_lab = cv2.pyrUp(result_lab, dstsize=size)
        result_lab = cv2.add(result_lab, blended_pyr[i])

    # Convert back to BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Final refinement with Poisson blending
    overlap = (mask1 > 0.5) & (mask2 > 0.5)
    if np.sum(overlap) > 2000:
        overlap_mask = overlap.astype(np.uint8) * 255
        y, x = np.nonzero(overlap)
        if len(x) > 0 and len(y) > 0:
            center = (int(np.median(x)), int(np.median(y)))
            try:
                result = cv2.seamlessClone(warped2, result, overlap_mask, center, cv2.MIXED_CLONE)
            except cv2.error:
                pass

    return result


def create_panorama(images, num_features=1000):
    if len(images) < 2:
        return images[0]

    panorama = images[0]

    for i in range(1, len(images)):
        # Use existing functions
        img_pair = [panorama, images[i]]
        corners, responses = detect_corners(img_pair, num_features)

        all_anms_corners = []
        for corners_i, response_i in zip(corners, responses):
            anms_corners = adaptive_non_maximal_suppression(response_i, corners_i, num_features)
            all_anms_corners.append(anms_corners)

        descriptors = []
        valid_corners = []
        for j, img in enumerate(img_pair):
            desc, v_corners = extract_features(img, all_anms_corners[j])
            descriptors.append(desc)
            valid_corners.append(v_corners)

        matches = match_features(descriptors[0], descriptors[1], valid_corners[0], valid_corners[1])

        if len(matches) < 10:
            print(f"Skipping image {i} - insufficient matches ({len(matches)})")
            continue

        H, mask = ransac_homography(matches)
        if H is None:
            print(f"Skipping image {i} - homography estimation failed")
            continue

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