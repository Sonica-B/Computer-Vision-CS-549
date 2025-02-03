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
import torch
import argparse
import os
from Network.Network import HomographyModel
# Add any python libraries here

def estimate_homography_deep(img1, img2, model, patch_size=(128, 128)):
    """
    Estimate homography between two images using deep learning model
    """
    device = next(model.parameters()).device
    # Resize images to 128x128
    img1_resized = cv2.resize(img1, patch_size)
    img2_resized = cv2.resize(img2, patch_size)

    # Convert to RGB and normalize
    img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB) / 255.0
    img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB) / 255.0

    # Convert to torch tensors
    img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Stack images for network input
    x = torch.cat([img1_tensor, img2_tensor], dim=1)

    # Get prediction
    with torch.no_grad():
        H4pt = model(x)

    # Convert to full homography matrix using model's DLT layer
    corners = torch.tensor([
        [0, 0],
        [patch_size[0], 0],
        [patch_size[0], patch_size[1]],
        [0, patch_size[1]]
    ]).float().reshape(1, 8)

    H = model.tensor_DLT(H4pt, corners)

    # Scale homography to original image size
    scale_matrix = np.array([
        [img1.shape[1] / patch_size[0], 0, 0],
        [0, img1.shape[0] / patch_size[1], 0],
        [0, 0, 1]
    ])
    inv_scale_matrix = np.array([
        [patch_size[0] / img1.shape[1], 0, 0],
        [0, patch_size[1] / img1.shape[0], 0],
        [0, 0, 1]
    ])

    H = scale_matrix @ H.squeeze().cpu().numpy() @ inv_scale_matrix

    return H


def blend_images(img1, img2, H):
    """
    Blend two images using estimated homography
    """
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create panorama canvas
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners, H)

    [xmin, ymin] = np.int32(corners_transformed.min(axis=0).ravel())
    [xmax, ymax] = np.int32(corners_transformed.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation[0]],
                              [0, 1, translation[1]],
                              [0, 0, 1]])

    output_size = (xmax - xmin, ymax - ymin)

    # Warp and blend images
    warped1 = cv2.warpPerspective(img1, H_translation @ H, output_size)
    warped2 = cv2.warpPerspective(img2, H_translation, output_size)

    # Create masks for blending
    mask1 = (warped1 != 0).astype(np.float32)
    mask2 = (warped2 != 0).astype(np.float32)

    # Blend in overlapping regions
    overlap = mask1 * mask2
    panorama = (warped1 * (1 - overlap) + warped2 * overlap).astype(np.uint8)

    return panorama

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/model.ckpt',
                        help='Path to load trained model')
    Parser.add_argument('--InputPath', dest='InputPath', default='../Data/Test',
                        help='Path to input images')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    InputPath = Args.InputPath

    """
    Read a set of images for Panorama stitching
    """
    image_files = sorted([f for f in os.listdir(InputPath) if f.endswith(('.jpg', '.png'))])
    if len(image_files) < 2:
        print('Need at least 2 images')
        return

    images = []
    for img_file in image_files:
        img_path = os.path.join(InputPath, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    # Loading Unsupervised Model
    model = HomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()

    # Initialize panorama with first image
    panorama = images[0]

    # Stitch images sequentially
    for i in range(1, len(images)):
        # Estimate homography between current image pair
        H = estimate_homography_deep(images[i], panorama, model)

        # Blend images
        panorama = blend_images(images[i], panorama, H)

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    cv2.imwrite('mypano.png', panorama)
    print('Saved panorama as mypano.png')

if __name__ == "__main__":
    main()
