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
from Triangulation.DisambiguateCameraPose import DisambiguateCameraPose
from Triangulation.NonlinearTriangulation import NonlinearTriangulation

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

# Calculating F matrices
matches = {}
for i in range(len(correspondences) + 1):
    for j in range(i, len(correspondences) + 1):
        if i != j:
            key = str(i+1) + "_" + str(j+1)
            matches[key] = [[], []]

for i in range(len(correspondences)):
    for j in correspondences[i]:
        img1 = i + 1
        img2 = j[2]
        matches[str(img1) + "_" + str(img2)][0].append(j[0])
        matches[str(img1) + "_" + str(img2)][1].append(j[1])

print("1. Estimating Fundamental Matrices...")
F = []
su = 0
for key in tqdm(matches):
    x = np.array(matches[key][0])
    x_ = np.array(matches[key][1])
    su += np.shape(x)[0]
    F.append(estimate_fundamental_matrix(x, x_))


print("2. Applying RANSAC for outlier rejection...")
# Applying RANSAC algorithm
F_inlier = []
inliers = []
for key in tqdm(matches):
    x = np.array(matches[key][0])
    x_ = np.array(matches[key][1])
    f, inl = InlierRANSAC(x, x_)
    F_inlier.append(f)
    inliers.append(inl)



for idx, key in enumerate(matches):
    print(f"Image pair {key}:")
    print(f"Total matches: {len(matches[key][0])}")
    print(f"Number of inliers: {len(inliers[idx])}")
    print("---")

# Estimate Essential Matrix from F_Inliers Fundamental Matrix
print("3. Computing Essential Matrices...")
E = []
for i in tqdm(range(len(F_inlier))):
    Fundamental_matrix = F_inlier[i]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    E.append(EssentialMatrixFromFundamentalMatrix(Fundamental_matrix, K))

# Extract Camera Poses ( 4 for each image pair)
print("4. Extracting Camera Poses...")
Camera_centers = []
Camera_rotations = []
for i in tqdm(range(len(E))):
    cen, rot = ExtractCameraPoses(E[i])
    Camera_centers.append(cen)
    Camera_rotations.append(rot)


for idx, key in enumerate(matches):
    print(f"\nCamera poses for pair {key}:")
    print("Centers:")
    for c in Camera_centers[idx]:
        print(c.reshape(-1))
    print("Determinants of rotation matrices:")
    for r in Camera_rotations[idx]:
        print(np.linalg.det(r))


# Traingulation Check for Cheirality Condition

print("5. Performing Triangulation...")
# Initialize containers for 3D points and camera poses
all_3D_points = {}
final_camera_poses = {}

# Set first camera as reference (at origin)
C1 = np.zeros((3, 1))
R1 = np.eye(3)
final_camera_poses['1'] = (C1, R1)

# Process each image pair
for idx, key in enumerate(tqdm(matches)):
    img1_idx, img2_idx = key.split('_')
    x = np.array(matches[key][0])
    x_ = np.array(matches[key][1])

    # Get inlier points
    inlier_indices = inliers[idx]
    x_inliers = x[inlier_indices]
    x__inliers = x_[inlier_indices]

    # Normalize image points
    x_inliers_norm = x_inliers / np.array([fx, fy])
    x__inliers_norm = x__inliers / np.array([fx, fy])

    # Disambiguate camera pose
    C_best, R_best, X_initial = DisambiguateCameraPose(
        K,
        Camera_centers[idx],
        Camera_rotations[idx],
        x_inliers_norm,
        x__inliers_norm
    )

    # Scale the initial reconstruction
    scale_factor = 1.0  # Adjust this based on your scene
    X_initial = X_initial * scale_factor
    C_best = C_best * scale_factor

    # Refine 3D points using non-linear optimization
    X_refined = NonlinearTriangulation(
        K,
        C1, R1,
        C_best, R_best,
        x_inliers,
        x__inliers,
        X_initial
    )

    # Store results
    final_camera_poses[img2_idx] = (C_best, R_best)
    all_3D_points[key] = X_refined


# Visualize 3D reconstruction
print("6. Visualizing Results...")
plot_3d_points_and_cameras(all_3D_points, final_camera_poses,
                          save_path="Phase1/Outputs/reconstruction_3d.png")