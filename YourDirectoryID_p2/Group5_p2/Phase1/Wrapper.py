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
from Triangulation.LinearTriangulation import LinearTriangulation
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
file = f"/Users/tvidk/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/WPI/Education/RBE 549/Project 2/Divam/YourDirectoryID_p2/Phase1/P2Data/"
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

F = []
su = 0
for key in tqdm(matches):
    x = np.array(matches[key][0])
    x_ = np.array(matches[key][1])
    su += np.shape(x)[0]
    F.append(estimate_fundamental_matrix(x, x_))

# Applying RANSAC algorithm
F_inlier = []
inliers = []
for key in tqdm(matches):
    x = np.array(matches[key][0])
    x_ = np.array(matches[key][1])
    f, inl = InlierRANSAC(x, x_)
    F_inlier.append(f)
    inliers.append(inl)

# Estimate Essential Matrix from F_Inliers Fundamental Matrix

E = []
for i in tqdm(range(len(F_inlier))):
    Fundamental_matrix = F_inlier[i]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    E.append(EssentialMatrixFromFundamentalMatrix(Fundamental_matrix, K))

# Extract Camera Poses ( 4 for each image pair)
Camera_centers = []
Camera_rotations = []
for i in tqdm(range(len(E))):
    cen, rot = ExtractCameraPoses(E[i])
    Camera_centers.append(cen)
    Camera_rotations.append(rot)

# Traingulation Check for Cheirality Condition
