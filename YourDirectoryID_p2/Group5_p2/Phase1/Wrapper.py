import numpy as np
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *
from plotting import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from Triangulation.LinearTriangulation import *
from Triangulation.DisambiguateCameraPose import *
from Triangulation.NonlinearTriangulation import *

# From Calibration.txt file
fx = 531.122155322710
fy = 531.541737503901
cx = 407.192550839899
cy = 313.308715048366


K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

def extract_correspondences(matching_file):
    correspondences = []
    rgb = []
    with open(matching_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            data = list(map(float, line.split()))
            num_matches = int(data[0])
            u_current, v_current = data[4], data[5]
            for i in range(num_matches-1):
                idx = 6 + i * 3
                image_id = int(data[idx])
                u_match, v_match = data[idx + 1], data[idx + 2]
                correspondences.append([(u_current, v_current), (u_match, v_match), image_id])
                rgb.append([data[1], data[2], data[3]])
    return correspondences, rgb

# Reading the matching_i.txt files and storing the values
file = f"D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p2\\Group5_p2\\P2Data\\"
correspondences = []
rgb = []
for i in range(1, 5):
    c, r = extract_correspondences(f"{file}matching{i}.txt")
    correspondences.append(c)
    rgb.append(r)

# Organize matches
matches = {}
for i in range(len(correspondences) + 1):
    for j in range(i, len(correspondences) + 1):
        if i != j:
            key = f"{i + 1}_{j + 1}"
            matches[key] = [[], []]

for i in range(len(correspondences)):
    for j in correspondences[i]:
        img1 = i + 1
        img2 = j[2]
        matches[f"{img1}_{img2}"][0].append(j[0])
        matches[f"{img1}_{img2}"][1].append(j[1])

print("1. Estimating Fundamental Matrices...")
F_matrices = []
for key in tqdm(matches):
    pts1 = np.array(matches[key][0])
    pts2 = np.array(matches[key][1])
    F_matrices.append(estimate_fundamental_matrix(pts1, pts2))

# Visualize initial matches
print("Visualizing initial feature matches...")
for key in matches:
    img1 = f"{file}{key.split('_')[0]}.png"
    img2 = f"{file}{key.split('_')[1]}.png"
    pts1 = np.array(matches[key][0])
    pts2 = np.array(matches[key][1])
    plot_correspondences(img1, img2, pts1, pts2, key)

print("2. Applying RANSAC for outlier rejection...")
F_inlier = []
inliers = []
for key in tqdm(matches):
    pts1 = np.array(matches[key][0])
    pts2 = np.array(matches[key][1])
    f, inl = InlierRANSAC(pts1, pts2)
    F_inlier.append(f)
    inliers.append(inl)

# Visualize RANSAC results
print("Visualizing RANSAC inliers...")
for i, key in enumerate(matches):
    pts1 = np.array(matches[key][0])
    pts2 = np.array(matches[key][1])
    img1 = f"{file}{key.split('_')[0]}.png"
    img2 = f"{file}{key.split('_')[1]}.png"
    plot_inliers(img1, img2, pts1, pts2, inliers[i], key)

# Print match statistics
for idx, key in enumerate(matches):
    print(f"Image pair {key}:")
    print(f"Total matches: {len(matches[key][0])}")
    print(f"Number of inliers: {len(inliers[idx])}")
    print("---")

print("3. Computing Essential Matrices...")
E_matrices = []
for i in tqdm(range(len(F_inlier))):
    E = EssentialMatrixFromFundamentalMatrix(F_inlier[i], K)
    E_matrices.append(E)

print("4. Extracting Camera Poses...")
Camera_centers = []
Camera_rotations = []
for i in tqdm(range(len(E_matrices))):
    centers, rotations = ExtractCameraPoses(E_matrices[i])
    Camera_centers.append(centers)
    Camera_rotations.append(rotations)

# Print camera poses
for idx, key in enumerate(matches):
    print(f"\nCamera poses for pair {key}:")
    print("Centers:")
    for c in Camera_centers[idx]:
        print(c.reshape(-1))
    print("Determinants of rotation matrices:")
    for r in Camera_rotations[idx]:
        print(np.linalg.det(r))

print("5. Performing Triangulation...")
# Initialize containers for 3D points and camera poses
all_3D_points = {}
final_camera_poses = {}

# Set first camera as reference
C1 = np.zeros(3)
R1 = np.eye(3)
final_camera_poses['1'] = (C1, R1)

# Process each image pair
for idx, key in enumerate(tqdm(matches)):
    img1_idx, img2_idx = key.split('_')
    pts1 = np.array(matches[key][0])
    pts2 = np.array(matches[key][1])

    # Get inlier points
    inlier_indices = inliers[idx]
    pts1_inliers = pts1[inlier_indices]
    pts2_inliers = pts2[inlier_indices]

    # Get all four possible reconstructions for visualization
    all_points_3d = []
    for i in range(4):
        X = LinearTriangulation(K, C1, R1,
                                Camera_centers[idx][i], Camera_rotations[idx][i],
                                pts1_inliers, pts2_inliers)
        all_points_3d.append(X)

    # Plot initial triangulation with all four possible poses
    print(f"\nSaving initial four configurations for pair {key}")
    plot_initial_triangulation_with_poses(all_points_3d, Camera_centers[idx], Camera_rotations[idx])

    # Disambiguate camera pose
    C_best, R_best, X_initial = DisambiguateCameraPose(
        K, Camera_centers[idx], Camera_rotations[idx],
        pts1_inliers, pts2_inliers
    )

    if X_initial is None:
        print(f"Skipping pair {key} due to failed triangulation")
        continue

    # Refine 3D points using non-linear optimization
    X_refined = NonlinearTriangulation(
        K, C1, R1, C_best, R_best,
        pts1_inliers, pts2_inliers, X_initial
    )

    # Compare linear vs nonlinear triangulation results
    print(f"Saving linear vs non-linear triangulation comparison for pair {key}")
    compare_linear_nonlinear_triangulation(X_initial, X_refined, idx)

    # Get reprojections for visualization
    proj_pts1_linear = project_points(X_initial, K, C1, R1)
    proj_pts2_linear = project_points(X_initial, K, C_best, R_best)
    proj_pts1_nonlinear = project_points(X_refined, K, C1, R1)
    proj_pts2_nonlinear = project_points(X_refined, K, C_best, R_best)

    # Plot reprojection comparison
    print(f"Saving reprojection comparison for pair {key}")
    img1_path = f"{file}{key.split('_')[0]}.png"
    img2_path = f"{file}{key.split('_')[1]}.png"
    plot_reprojection_comparison(img1_path, img2_path, pts1_inliers, pts2_inliers,
                                 proj_pts1_linear, proj_pts2_linear,
                                 proj_pts1_nonlinear, proj_pts2_nonlinear, idx)

    # Store results
    final_camera_poses[img2_idx] = (C_best, R_best)
    all_3D_points[key] = X_refined

# Visualize 3D reconstruction
print("6. Visualizing Results...")
plot_3d_points_and_cameras(all_3D_points, final_camera_poses,
                          save_path="Outputs/reconstruction_3d.png")