"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def tensor_DLT(delta, corners):
    """Convert 4-point parametrization to homography matrix using DLT"""
    batch_size = delta.shape[0]
    device = delta.device

    # Reshape corners and deltas
    corners = corners.view(batch_size, 4, 2)
    delta = delta.view(batch_size, 4, 2)
    dst = corners + delta
    src = corners

    # Construct DLT matrix A
    A = torch.zeros(batch_size, 8, 9, device=device)
    for i in range(4):
        x, y = src[:, i, 0], src[:, i, 1]
        u, v = dst[:, i, 0], dst[:, i, 1]
        A[:, i * 2] = torch.stack([x, y, torch.ones_like(x),
                               torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x),
                               -x * u, -y * u, -u], dim=1)
        A[:, i * 2 + 1] = torch.stack([torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x),
                                   x, y, torch.ones_like(x),
                                   -x * v, -y * v, -v], dim=1)

    # Solve using SVD
    _, _, V = torch.svd(A)
    H = V[:, :, -1].view(batch_size, 9)
    H = H / H[:, 8].view(-1, 1)
    H = H.view(batch_size, 3, 3)
    return H


def create_grid(h, w, device):
    """Create normalized grid coordinates"""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    grid = torch.stack([x, y], dim=2).float()
    grid[:, :, 0] = 2.0 * grid[:, :, 0] / (w - 1) - 1.0
    grid[:, :, 1] = 2.0 * grid[:, :, 1] / (h - 1) - 1.0
    return grid


def bilinear_sample(img, grid):
    """Differentiable bilinear sampling"""
    B, C, H, W = img.shape

    # Normalize coordinates to [0, W-1/H-1]
    x = ((grid[..., 0] + 1) / 2) * (W - 1)
    y = ((grid[..., 1] + 1) / 2) * (H - 1)

    # Get integer and fractional parts
    x0 = torch.floor(x).long().clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # Get corner weights
    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(1)
    wb = ((x1.float() - x) * (y - y0.float())).unsqueeze(1)
    wc = ((x - x0.float()) * (y1.float() - y)).unsqueeze(1)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(1)

    # Get linear indices
    idx_a = (y0 * W + x0).clamp(0, H * W - 1)
    idx_b = (y0 * W + x1).clamp(0, H * W - 1)
    idx_c = (y1 * W + x0).clamp(0, H * W - 1)
    idx_d = (y1 * W + x1).clamp(0, H * W - 1)

    # Gather and weight pixels
    img_flat = img.view(B, C, -1)
    output = (wa * torch.gather(img_flat, 2, idx_a.unsqueeze(1).expand(-1, C, -1)) +
              wb * torch.gather(img_flat, 2, idx_b.unsqueeze(1).expand(-1, C, -1)) +
              wc * torch.gather(img_flat, 2, idx_c.unsqueeze(1).expand(-1, C, -1)) +
              wd * torch.gather(img_flat, 2, idx_d.unsqueeze(1).expand(-1, C, -1)))

    return output.view(B, C, H, W)
# def LossFn(delta, img_a, patch_b, corners):
#     ###############################################
#     # Fill your loss function of choice here!
#     ###############################################
#     # Get full homography matrix using DLT
#     batch_size = delta.shape[0]
#     H = tensor_DLT(delta, corners)
#
#     # Create sampling grid
#     h, w = patch_b.shape[-2:]
#     grid = create_grid(h, w, delta.device)
#
#     # Transform grid with homography
#     points = grid.reshape(-1, 2)
#     points = torch.cat([points, torch.ones(points.shape[0], 1, device=delta.device)], dim=1)
#     warped_points = torch.matmul(H, points.t().unsqueeze(0))
#     warped_points = warped_points.permute(0, 2, 1)
#     warped_points = warped_points / (warped_points[..., 2:] + 1e-8)
#     warped_points = warped_points[..., :2]
#     warped_grid = warped_points.reshape(batch_size, h, w, 2)
#
#     # Sample from source image
#     warped_a = bilinear_sample(img_a, warped_grid)
#
#     ###############################################
#     # You can use kornia to get the transform and warp in this project
#     # Bonus if you implement it yourself
#     ###############################################
#     loss = torch.mean(torch.abs(warped_a - patch_b)) #Calculate L1 photometric loss
#     return loss

def LossFn(delta, img_a, patch_b, corners):
    # Compute homography
    H = tensor_DLT(delta, corners)

    # Create sampling grid
    h, w = patch_b.shape[-2:]
    grid = create_grid(h, w, delta.device)
    grid = grid.unsqueeze(0).repeat(delta.shape[0], 1, 1, 1)

    # Transform grid
    ones = torch.ones(*grid.shape[:-1], 1, device=delta.device)
    points = torch.cat([grid, ones], dim=-1)
    points_t = points.reshape(delta.shape[0], -1, 3).transpose(1, 2)
    warped_points = torch.bmm(H, points_t)
    warped_points = warped_points.transpose(1, 2).reshape(delta.shape[0], h, w, 3)
    warped_grid = warped_points[..., :2] / (warped_points[..., 2:] + 1e-8)

    # Sample and compute loss
    warped_a = bilinear_sample(img_a, warped_grid)
    loss = torch.mean(torch.abs(warped_a - patch_b))
    return loss

class Net(nn.Module):
    def __init__(self, InputSize, OutputSize):
        super().__init__()

        # Input: [B,6,128,128]
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # [B,64,128,128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # [B,64,64,64]
        )

        # Adjusted STN layers for correct dimensions
        self.localization = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # [B,32,64,64]
            nn.MaxPool2d(2),  # [B,32,32,32]
            nn.ReLU(True),
            nn.Conv2d(32, 20, 3, padding=1),  # [B,20,32,32]
            nn.MaxPool2d(2),  # [B,20,16,16]
            nn.ReLU(True)
        )

        # 20 * 16 * 16 = 5120
        self.fc_loc = nn.Sequential(
            nn.Linear(5120, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 256 * 32 * 32 = 262144
        self.regressor = nn.Sequential(
            nn.Linear(262144, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, OutputSize)
        )

        # Initialize STN weights
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 20 * 16 * 16)  # Modified to match actual tensor size
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, xa, xb):
        x = torch.cat([xa, xb], dim=1)
        print("Concatenated tensor shape:", x.shape)
        x = self.features(x)
        x = self.stn(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        out = self.regressor(x)
        return out


class HomographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Net(InputSize=6 * 128 * 128, OutputSize=8)

    def forward(self, patch_a, patch_b):
        x = torch.cat([patch_a, patch_b], dim=1)
        return self.model(x)

    def training_step(self, batch):
        patch_a = batch[1]  # [B,3,H,W]
        patch_b = batch[2]  # [B,3,H,W]
        corners = batch[3]  # [B,8]

        delta = self.forward(patch_a, patch_b)
        loss = LossFn(delta, patch_a, patch_b, corners)
        return {'loss': loss}

    def validation_step(self, batch):
        patch_a = batch[1]  # [B,3,H,W]
        patch_b = batch[2]  # [B,3,H,W]
        corners = batch[3]  # [B,8]

        delta = self.forward(patch_a, patch_b)
        loss = LossFn(delta, patch_a, patch_b, corners)
        return {'val_loss': loss}