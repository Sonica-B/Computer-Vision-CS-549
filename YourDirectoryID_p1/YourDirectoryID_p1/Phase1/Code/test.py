import numpy as np
import cv2
import argparse
import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur


def detect_corners(images, num_features, output_dir):
    """
    Detect corners using cv2.goodFeaturesToTrack and Gaussian filtering
    """
    features = {}

    for idx, img in enumerate(images):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Corner detection parameters
        feature_params = dict(
            maxCorners=num_features,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=5
        )

        # Detect corners
        corners = cv2.goodFeaturesToTrack(blurred, **feature_params)
        corners = np.int0(corners)

        # Draw corners
        img_corners = img.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_corners, (x, y), 3, (0, 0, 255), -1)

        # Save visualization
        output_path = os.path.join(output_dir, f'corners_{idx}.jpg')
        cv2.imwrite(output_path, img_corners)

        # Store corner coordinates
        features[idx] = corners

        print(f"Image {idx}: Found {len(corners)} corners")

    return features


def ANMS(features, num_best, images, output_dir):
    """
    Adaptive Non-Maximal Suppression: Optimizes corner distribution
    """
    selected_features = {}

    for idx, corners in features.items():
        gray = cv2.cvtColor(images[idx], cv2.COLOR_BGR2GRAY)
        corner_img = cv2.cornerHarris(gray, 2, 3, 0.04)

        N_strong = len(corners)
        r = np.inf * np.ones(N_strong)

        for i in range(N_strong):
            for j in range(N_strong):
                y_i, x_i = corners[i][0]  # Note: corners are in (y,x) format
                y_j, x_j = corners[j][0]

                # Compare corner strengths and update r if necessary
                if corner_img[x_j, y_j] > corner_img[x_i, y_i]:
                    ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
                    r[i] = min(r[i], ED)

        # Select fewer corners than detected for better distribution
        num_corners = min(num_best, N_strong)
        idx_strong = np.argsort(-r)[:num_corners]
        selected_corners = corners[idx_strong].reshape(-1, 2)

        # Visualization
        img_viz = images[idx].copy()
        for corner in selected_corners:
            cv2.circle(img_viz, (int(corner[0]), int(corner[1])), 3, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(output_dir, f'anms_{idx}.jpg'), img_viz)

        selected_features[idx] = selected_corners

    return selected_features


def get_feature_descriptor(img, keypoint):
    """
    Generate feature descriptor for a keypoint
    """
    x, y = int(keypoint[0]), int(keypoint[1])

    # Get 41x41 patch centered at keypoint
    half_size = 20
    if (x - half_size < 0 or x + half_size + 1 > img.shape[1] or
            y - half_size < 0 or y + half_size + 1 > img.shape[0]):
        return None

    patch = img[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(patch, (3, 3), 0)

    # Subsample to 8x8
    subsampled = cv2.resize(blurred, (8, 8))

    # Reshape to 64x1 vector
    feature_vector = subsampled.reshape(64)

    # Standardize to zero mean and unit variance
    mean = np.mean(feature_vector)
    std = np.std(feature_vector)
    if std != 0:
        feature_vector = (feature_vector - mean) / std

    return feature_vector


def extract_features(images, selected_features, output_dir):
    """
    Extract feature descriptors and save visualizations
    """
    all_descriptors = {}

    for idx, keypoints in selected_features.items():
        gray = cv2.cvtColor(images[idx], cv2.COLOR_BGR2GRAY)

        descriptors = []
        valid_keypoints = []

        # Visualization image
        img_viz = images[idx].copy()

        for kp in keypoints:
            desc = get_feature_descriptor(gray, kp)
            if desc is not None:
                descriptors.append(desc)
                valid_keypoints.append(kp)

                # Draw patch boundary
                x, y = int(kp[0]), int(kp[1])
                cv2.rectangle(img_viz, (x - 20, y - 20), (x + 20, y + 20), (255, 0, 0), 1)

        # Save visualization
        cv2.imwrite(os.path.join(output_dir, f'features_{idx}.jpg'), img_viz)

        all_descriptors[idx] = {
            'keypoints': np.array(valid_keypoints),
            'descriptors': np.array(descriptors)
        }

    return all_descriptors


def match_features(descriptors, images, output_dir, ratio_threshold=0.8):
    """
    Match features between all possible image pairs excluding self-matches
    """
    matches_all = {}
    num_images = len(images)

    for i in range(num_images):
        for j in range(i + 1, num_images):
            desc1 = descriptors[i]['descriptors']
            desc2 = descriptors[j]['descriptors']
            kp1 = descriptors[i]['keypoints']
            kp2 = descriptors[j]['keypoints']

            matches = []
            for idx1, d1 in enumerate(desc1):
                distances = np.sum((desc2 - d1) ** 2, axis=1)
                idx2_sorted = np.argsort(distances)
                ratio = distances[idx2_sorted[0]] / distances[idx2_sorted[1]]

                if ratio < ratio_threshold:
                    matches.append((idx1, idx2_sorted[0]))

            matched_kp1 = []
            matched_kp2 = []
            for idx1, idx2 in matches:
                matched_kp1.append(kp1[idx1])
                matched_kp2.append(kp2[idx2])

            # Visualization
            img1, img2 = images[i], images[j]
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1 + w2] = img2

            for pt1, pt2 in zip(matched_kp1, matched_kp2):
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0] + w1), int(pt2[1])
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
                cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)

            cv2.imwrite(os.path.join(output_dir, f'matches_{i}_{j}.jpg'), vis)

            matches_all[f'{i}_{j}'] = {
                'matches': matches,
                'keypoints1': matched_kp1,
                'keypoints2': matched_kp2
            }

    return matches_all


def compute_homography(pts1, pts2):
    """Compute homography matrix between two sets of points"""
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.extend([
            [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2],
            [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]
        ])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H


def compute_ssd(pts1, pts2, H):
    """Compute Sum of Square Differences between points"""
    pts1_homog = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_proj = (H @ pts1_homog.T).T
    pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:]
    return np.sum((pts2_proj - pts2) ** 2, axis=1)


def ransac_homography(matches_dict, output_dir, num_iterations=1000, threshold=5.0, inlier_ratio=0.9):
    """
    Perform RANSAC to find best homography matrix
    """
    results = {}

    for pair_key, match_data in matches_dict.items():
        pts1 = np.array(match_data['keypoints1'])
        pts2 = np.array(match_data['keypoints2'])

        best_H = None
        best_inliers = []
        max_inliers = 0

        for _ in range(num_iterations):
            if len(pts1) < 4:
                continue

            indices = np.random.choice(len(pts1), 4, replace=False)
            sample_pts1 = pts1[indices]
            sample_pts2 = pts2[indices]

            H = compute_homography(sample_pts1, sample_pts2)
            ssd = compute_ssd(pts1, pts2, H)
            inliers = ssd < threshold
            num_inliers = np.sum(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_inliers = inliers
                best_H = H

            if num_inliers / len(pts1) > inlier_ratio:
                break

        if best_H is not None:
            inlier_pts1 = pts1[best_inliers]
            inlier_pts2 = pts2[best_inliers]
            final_H = compute_homography(inlier_pts1, inlier_pts2)

            results[pair_key] = {
                'H': final_H,
                'inliers': best_inliers,
                'num_inliers': max_inliers,
                'inlier_pts1': inlier_pts1,
                'inlier_pts2': inlier_pts2,
                'inlier_ratio': max_inliers / len(pts1)
            }

            np.save(os.path.join(output_dir, f'homography_{pair_key}.npy'), final_H)

    return results

def visualize_ransac_results(ransac_results, images, output_dir):
    """Visualize RANSAC matches and homography transformations"""
    for pair_key, result in ransac_results.items():
        img_idx1, img_idx2 = map(int, pair_key.split('_'))
        img1, img2 = images[img_idx1], images[img_idx2]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Draw matches
        vis_matches = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis_matches[:h1, :w1] = img1
        vis_matches[:h2, w1:w1+w2] = img2

        inlier_pts1 = result['inlier_pts1']
        inlier_pts2 = result['inlier_pts2']

        # Draw inlier matches
        for pt1, pt2 in zip(inlier_pts1, inlier_pts2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0] + w1), int(pt2[1])
            cv2.line(vis_matches, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(vis_matches, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(vis_matches, (x2, y2), 3, (0, 0, 255), -1)

        # Draw text with match statistics
        text = f"Inliers: {result['num_inliers']} ({result['inlier_ratio']:.2%})"
        cv2.putText(vis_matches, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(output_dir, f'ransac_vis_{pair_key}.jpg'), vis_matches)


def warp_and_blend(stitch_data, images):
    """Warps and blends images based on stitch sequence and homographies"""
    sequence = stitch_data['sequence']
    homographies = stitch_data['homographies']
    inlier_matches = stitch_data['matches']
    source_idx = sequence[0][0]

    height, width = images[source_idx].shape[:2]

    def compute_transformation():
        corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]])
        all_corners = [corners]

        H_cumulative = np.eye(3)
        for i in range(len(sequence)):
            H_cumulative = H_cumulative @ homographies[f"{sequence[i][0]}_{sequence[i][1]}"]
            warped_corners = H_cumulative @ corners.T
            warped_corners = warped_corners / warped_corners[2]
            all_corners.append(warped_corners.T)

        all_corners = np.vstack(all_corners)
        [xmin, ymin] = np.min(all_corners, axis=0)[:2]
        [xmax, ymax] = np.max(all_corners, axis=0)[:2]

        offset = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ])

        return offset @ H_cumulative, (int(ymax - ymin), int(xmax - xmin))

    def blend_images(warped1, warped2, seam_points):
        mask = np.zeros_like(warped1, dtype=np.float32)
        cv2.fillPoly(mask, [seam_points], (1, 1, 1))

        blended = np.where(mask == 1,
                           warped1 * 0.5 + warped2 * 0.5,
                           np.where(warped1.sum(axis=2, keepdims=True) != 0, warped1, warped2))
        return blended.astype(np.uint8)

    H_total, (height_pano, width_pano) = compute_transformation()
    final_panorama = cv2.warpPerspective(images[source_idx], H_total, (width_pano, height_pano))

    for i in range(len(sequence)):
        next_img_idx = sequence[i][1]
        pair_key = f"{sequence[i][0]}_{sequence[i][1]}"

        H = homographies[pair_key]
        warped_next = cv2.warpPerspective(images[next_img_idx], H, (width_pano, height_pano))

        inliers = inlier_matches[pair_key]['inliers']
        seam_points = np.int32(inliers.reshape(-1, 2))

        final_panorama = blend_images(final_panorama, warped_next, seam_points)

    # Crop black borders
    gray = cv2.cvtColor(final_panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    final_panorama = final_panorama[y:y + h, x:x + w]

    return final_panorama
def create_panorama(images,ransac_results, output_dir, min_matches=10):
    """Creates panorama and saves final image"""
    match_counts = {}
    for pair_key, result in ransac_results.items():
        i, j = map(int, pair_key.split('_'))
        count = result['num_inliers']
        match_counts[(i, j)] = count

    edges = [(i, j, count) for (i, j), count in match_counts.items() if count > min_matches]
    if not edges:
        print("No good matches found between images")
        return None

    used = set()
    sequence = []
    current = min(i for i, _, _ in edges)

    while True:
        used.add(current)
        next_best = None
        max_count = min_matches

        for i, j, count in edges:
            if i == current and j not in used and count > max_count:
                next_best = j
                max_count = count
            elif j == current and i not in used and count > max_count:
                next_best = i
                max_count = count

        if next_best is None:
            break

        sequence.append((current, next_best))
        current = next_best

    if len(sequence) < 1:
        print("Not enough matches to create panorama")
        return None

    stitch_data = {
        'sequence': sequence,
        'homographies': {pair_key: result['H'] for pair_key, result in ransac_results.items()},
        'matches': {pair_key: {'inliers': result['inlier_pts1'], 'ratio': result['inlier_ratio']}
                    for pair_key, result in ransac_results.items()}
    }

    # Save stitch data
    np.save(os.path.join(output_dir, 'panorama_data.npy'), stitch_data)

    # Generate final panorama
    final_panorama = warp_and_blend(stitch_data, images)

    # Save final image
    cv2.imwrite(os.path.join(output_dir, 'panorama.jpg'), final_panorama)

    return final_panorama

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumFeatures', default=1000, type=int,
                        help='Number of features to detect')
    parser.add_argument('--folder', default='../Data/Train/Set3',
                        help='Path to image folder')
    parser.add_argument('--output_dir', default='Outputs/Set3',
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

    # Detect corners
    features = detect_corners(images, args.NumFeatures, args.output_dir)
    #ANMS
    selected_features = ANMS(features, args.NumFeatures, images, args.output_dir)
    # Extract feature descriptors
    descriptors = extract_features(images, selected_features, args.output_dir)
    # Feature Matching
    matches = match_features(descriptors, images, args.output_dir)
    # RANSAC
    ransac_results = ransac_homography(matches, args.output_dir)
    visualize_ransac_results(ransac_results, images, args.output_dir)

    # #Pano
    # stitch_data = create_panorama(images,ransac_results, args.output_dir)



if __name__ == '__main__':
    main()