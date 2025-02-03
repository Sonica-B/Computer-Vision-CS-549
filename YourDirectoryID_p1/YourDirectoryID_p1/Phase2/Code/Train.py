#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import json
import torch.nn.functional as F
from torchvision.transforms.functional import resize

#
# def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
#     """
#     Inputs:
#     BasePath - Path to COCO folder without "/" at the end
#     DirNamesTrain - Variable with Subfolder paths to train files
#     NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
#     TrainCoordinates - Coordinatess corresponding to Train
#     NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
#     ImageSize - Size of the Image
#     MiniBatchSize is the size of the MiniBatch
#     Outputs:
#     I1Batch - Batch of images
#     CoordinatesBatch - Batch of coordinates
#     """
#     I1Batch = []
#     CoordinatesBatch = []
#
#     ImageNum = 0
#     while ImageNum < MiniBatchSize:
#         # Generate random image
#         RandIdx = random.randint(0, len(DirNamesTrain) - 1)
#
#         # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
#         # ImageNum += 1
#
#
#         JsonFile = DirNamesTrain[RandIdx]
#
#         # Load JSON file
#         JsonPath = os.path.join(BasePath, JsonFile + ".json")
#         with open(JsonPath, 'r') as f:
#             data = json.load(f)
#
#         # Randomly select an entry from the JSON file
#         RandEntry = random.choice(data)
#
#         # Extract the stacked_input1 (image) and H4pt_input2 (coordinates)
#         stacked_input1 = np.array(RandEntry['stacked_input1'], dtype=np.float32)
#         H4pt_input2 = np.array(RandEntry['H4pt_input2'], dtype=np.float32)
#
#         # Resize the image to (128, 128, 6)
#         stacked_input1_resized = resize(torch.from_numpy(stacked_input1).permute(2, 0, 1), ImageSize)
#         stacked_input1_resized = stacked_input1_resized.permute(1, 2, 0).numpy()
#
#
#         ##########################################################
#         # Add any standardization or data augmentation here!
#         ##########################################################
#         # I1 = np.float32(cv2.imread(RandImageName))
#         # Coordinates = TrainCoordinates[RandIdx]
#
#         # # Append All Images and Mask
#         # I1Batch.append(torch.from_numpy(I1))
#         # CoordinatesBatch.append(torch.tensor(Coordinates))
#         # Append to batches
#         I1Batch.append(torch.from_numpy(stacked_input1_resized))
#         CoordinatesBatch.append(torch.from_numpy(H4pt_input2))
#         ImageNum += 1
#     return torch.stack(I1Batch), torch.stack(CoordinatesBatch)

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    I1Batch = []
    CoordinatesBatch = []

    for _ in range(MiniBatchSize):
        JsonFile = DirNamesTrain[0]
        JsonPath = os.path.join(BasePath, JsonFile + ".json")

        with open(JsonPath, 'r') as f:
            data = json.load(f)

        RandKey = random.choice(list(data.keys()))
        RandEntry = data[RandKey]

        stacked_input1 = np.array(RandEntry['stacked_input1'], dtype=np.float32) / 255.0
        H4pt_input2 = np.array(RandEntry['H4pt_input2'], dtype=np.float32)

        # Convert to tensor with correct shape [C,H,W]
        stacked_input1_tensor = torch.from_numpy(stacked_input1).permute(2, 0, 1)

        # Resize to target size
        stacked_input1_tensor = F.interpolate(
            stacked_input1_tensor.unsqueeze(0),
            size=(ImageSize[0], ImageSize[1]),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        H4pt_tensor = torch.from_numpy(H4pt_input2)

        I1Batch.append(stacked_input1_tensor)
        CoordinatesBatch.append(H4pt_tensor)

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    # Model setup
    model = HomographyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Tensorboard
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        LossThisBatch = None  # Initialize LossThisBatch to None

        # Debug: Print the number of iterations per epoch
        print(f"Epoch {Epochs}: NumIterationsPerEpoch = {NumIterationsPerEpoch}")

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            )

            # Debug shapes
            print("Input batch shape:", I1Batch.shape)  # Should be [B, 6, 128, 128]

            # Move to device
            I1Batch = I1Batch.to(device)
            CoordinatesBatch = CoordinatesBatch.to(device)

            # Split into patches
            patch_a = I1Batch[:, :3]  # [B,3,H,W]
            patch_b = I1Batch[:, 3:]  # [B,3,H,W]
            corners = CoordinatesBatch  # Coordinates [B, 8]

            # Predict output with forward pass
            Batch = [None, patch_a, patch_b, CoordinatesBatch]
            #Batch = [I1Batch, patch_a, patch_b, corners]
            result = model.validation_step(Batch)
            LossThisBatch = result["val_loss"]

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.flush()

        # Save model every epoch
        if LossThisBatch is not None:  # Only save if LossThisBatch has been assigned a value
            SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
            torch.save(
                {
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": LossThisBatch,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
