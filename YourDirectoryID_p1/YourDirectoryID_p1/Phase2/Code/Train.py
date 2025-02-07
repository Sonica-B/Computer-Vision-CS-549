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
from torch import optim
from Network.Network import *
import json
import torch
import numpy as np
import gc, time
from itertools import islice
#from Network.Network import HomographyModel
from Network.Network import UnsupervisedHomographyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Modules Imported")


def GenerateBatch(JsonPath, MiniBatchSize):
    """
    Inputs:
    JsonPath - Path to the JSON file containing images and H4pt
    MiniBatchSize - Number of samples per batch

    Outputs:
    I1Batch - Batch of images (Tensor of shape [MiniBatchSize, 128, 128, 6])
    CoordinatesBatch - Batch of H4pt coordinates (Tensor of shape [MiniBatchSize, 8])
    """
    I1Batch = []
    I2Batch = []
    CoordinatesBatch = []

    # Load JSON file
    with open(JsonPath, 'r') as f:
        dataset = json.load(f)

    # Randomly select MiniBatchSize samples
    RandIdxs = np.random.choice(len(dataset), MiniBatchSize, replace=False)

    for idx in tqdm(RandIdxs):
        image = np.array(dataset[str(idx)]['stacked_input1'])  # Shape (128, 128, 6)
        imga = np.array(dataset[str(idx)]['stacked_input1'])[:, :, :3].astype(np.uint8)
        imgb = np.array(dataset[str(idx)]['stacked_input1'])[:, :, 3:].astype(np.uint8)
        imga = cv2.resize(imga, (128, 128), interpolation=cv2.INTER_LINEAR)
        imgb = cv2.resize(imgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        img1 = np.reshape(imga, (128, 128, 3))
        img2 = np.reshape(imgb, (128, 128, 3))
        # image = np.concatenate((imga, imgb), axis=-1)
        h4pt = np.array(dataset[str(idx)]['H4pt_input2'])  # Shape (8,)

        # Convert to PyTorch Tensors
        I1Batch.append(torch.tensor(img1, dtype=torch.float32))
        I2Batch.append(torch.tensor(img2, dtype=torch.float32))
        CoordinatesBatch.append(torch.tensor(h4pt, dtype=torch.float32))

    return torch.stack(I1Batch).to(device, non_blocking=True), torch.stack(I2Batch).to(device,
                                                                                       non_blocking=True), torch.stack(
        CoordinatesBatch).to(device, non_blocking=True)

# def UnSupGenerateBatch(JsonPath, MiniBatchSize, ModelType):
#     I1Batch = []
#     I2Batch = []
#     CoordinatesBatch = []
#
#     with open(JsonPath, 'r') as f:
#         dataset = json.load(f)
#
#     RandIdxs = np.random.choice(len(dataset), MiniBatchSize, replace=False)
#
#     for idx in tqdm(RandIdxs):
#         image = np.array(dataset[str(idx)]['stacked_input1'])
#         imga = image[:, :, :3].astype(np.uint8)
#         imgb = image[:, :, 3:].astype(np.uint8)
#         imga = cv2.resize(imga, (128, 128), interpolation=cv2.INTER_LINEAR)
#         imgb = cv2.resize(imgb, (128, 128), interpolation=cv2.INTER_LINEAR)
#         img1 = np.reshape(imga, (128, 128, 3))
#         img2 = np.reshape(imgb, (128, 128, 3))
#
#         if ModelType == 'Sup':
#             h4pt = np.array(dataset[str(idx)]['H4pt_input2'])
#         else:
#             # Original corners for 128x128 patch
#             h4pt = np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 128.0], [0.0, 128.0]], dtype=np.float32).flatten()
#
#         I1Batch.append(torch.tensor(img1, dtype=torch.float32))
#         I2Batch.append(torch.tensor(img2, dtype=torch.float32))
#         CoordinatesBatch.append(torch.tensor(h4pt, dtype=torch.float32))
#
#     return (
#         torch.stack(I1Batch).to(device, non_blocking=True),
#         torch.stack(I2Batch).to(device, non_blocking=True),
#         torch.stack(CoordinatesBatch).to(device, non_blocking=True)
#     )
def read_json_in_chunks(file_path, chunk_size):
    with open(file_path, 'r') as f:
        data = {}
        for _ in range(chunk_size):
            try:
                line = next(f).strip()
                if line.startswith('{'):
                    obj = json.loads(line + '}')
                    data.update(obj)
            except StopIteration:
                break
            except json.JSONDecodeError:
                continue
        return data


# def UnSupGenerateBatch(JsonPath, MiniBatchSize, ModelType):
#     with open(JsonPath, 'r') as f:
#         dataset = json.load(f)
#
#     indices = np.random.choice(len(dataset), MiniBatchSize, replace=True)
#
#     I1Batch = []
#     I2Batch = []
#     CoordinatesBatch = []
#
#     for idx in indices:
#         # Load stacked image and split into img1/img2
#         image = np.array(dataset[str(idx)]['stacked_input1'])
#
#         # Extract img1 and img2 as uint8 (original data format)
#         img1_uint8 = image[:, :, :3].astype(np.uint8)
#         img2_uint8 = image[:, :, 3:].astype(np.uint8)
#
#         # Resize to 128x128 using OpenCV
#         img1_resized = cv2.resize(img1_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)
#         img2_resized = cv2.resize(img2_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)
#
#         # Normalize to [0, 1] and convert to float32
#         img1 = img1_resized.astype(np.float32) / 255.0
#         img2 = img2_resized.astype(np.float32) / 255.0
#
#         # Convert to PyTorch tensors [C, H, W]
#         img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
#         img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
#
#         # Original corners for 128x128 images
#         corners = np.array([[0, 0], [128, 0], [128, 128], [0, 128]], dtype=np.float32).flatten()
#
#         I1Batch.append(img1_tensor)
#         I2Batch.append(img2_tensor)
#         CoordinatesBatch.append(torch.from_numpy(corners))
#
#     return torch.stack(I1Batch), torch.stack(I2Batch), torch.stack(CoordinatesBatch)

def UnSupGenerateBatch(JsonPath, MiniBatchSize, ModelType):
    with open(JsonPath, 'r') as f:
        dataset = json.load(f)

    indices = np.random.choice(len(dataset), MiniBatchSize, replace=True)

    I1Batch = []
    I2Batch = []
    CoordinatesBatch = []

    for idx in indices:
        # Load stacked image and split into img1/img2
        image = np.array(dataset[str(idx)]['stacked_input1'])

        # Extract img1 and img2 as uint8 (original data format)
        img1_uint8 = image[:, :, :3].astype(np.uint8)
        img2_uint8 = image[:, :, 3:].astype(np.uint8)

        # Resize to 128x128 using OpenCV
        img1_resized = cv2.resize(img1_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)
        img2_resized = cv2.resize(img2_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and convert to float32
        img1 = img1_resized.astype(np.float32) / 255.0
        img2 = img2_resized.astype(np.float32) / 255.0

        # Convert to PyTorch tensors [C, H, W]
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()

        # Concatenate img1 and img2 along channel dimension
        img_stacked = torch.cat((img1_tensor, img2_tensor), dim=0)  # Shape: (6, 128, 128)

        # Extract real corner points from JSON (instead of fixed values)
        corners = np.array(dataset[str(idx)]['cornersA']).flatten().astype(np.float32)

        I1Batch.append(img_stacked)
        CoordinatesBatch.append(torch.from_numpy(corners))

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


# def TrainOperation(
#     DirNamesTrain,
#     TrainCoordinates,
#     NumTrainSamples,
#     ImageSize,
#     NumEpochs,
#     MiniBatchSize,
#     SaveCheckPoint,
#     CheckPointPath,
#     DivTrain,
#     LatestFile,
#     BasePath,
#     LogsPath,
#     ModelType,
#     JsonPath
# ):
#     """
#     Inputs:
#     ImgPH is the Input Image placeholder
#     DirNamesTrain - Variable with Subfolder paths to train files
#     TrainCoordinates - Coordinates corresponding to Train/Test
#     NumTrainSamples - length(Train)
#     ImageSize - Size of the image
#     NumEpochs - Number of passes through the Train data
#     MiniBatchSize is the size of the MiniBatch
#     SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
#     CheckPointPath - Path to save checkpoints/model
#     DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
#     LatestFile - Latest checkpointfile to continue training
#     BasePath - Path to COCO folder without "/" at the end
#     LogsPath - Path to save Tensorboard Logs
#         ModelType - Supervised or Unsupervised Model
#     Outputs:
#     Saves Trained network in CheckPointPath and Logs to LogsPath
#     """
#     # Predict output with forward pass
#     #model = HomographyModel().to(device)
#
#     if ModelType == 'Sup':
#         model = HomographyModel().to(device)
#     elif ModelType == 'Unsup':
#         model = UnsupervisedHomographyModel().to(device)
#
#     result = None
#     latest_loss = None
#
#     ###############################################
#     # Fill your optimizer of choice here!
#     ###############################################
#     Optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # Tensorboard
#     # Create a summary to monitor loss tensor
#     Writer = SummaryWriter(LogsPath)
#
#     if LatestFile is not None:
#         CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
#         # Extract only numbers from the name
#         StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
#         model.load_state_dict(CheckPoint["model_state_dict"])
#         print("Loaded latest checkpoint with the name " + LatestFile + "....")
#     else:
#         StartEpoch = 0
#         print("New model initialized....")
#
#     for Epochs in tqdm(range(StartEpoch, NumEpochs)):
#         print("Epoch " + str(Epochs))
#         NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
#         for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None
#             gc.collect()
#             I1Batch, I2Batch, CoordinatesBatch = GenerateBatch(
#                 JsonPath, MiniBatchSize
#             )
#             print("Batch Generated")
#
#             # Predict output with forward pass
#             Batch = 0, I1Batch, I2Batch, CoordinatesBatch, 0
#             # Training step
#             if ModelType == "Sup":
#                 PredicatedCoordinatesBatch = model(I1Batch, I2Batch)
#                 LossThisBatch = LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)
#                 result = {"loss": LossThisBatch, "val_loss": LossThisBatch}
#             else:
#                 result = model.training_step(Batch, PerEpochCounter)
#                 LossThisBatch = result["loss"]
#
#             Optimizer.zero_grad()
#             LossThisBatch.backward()
#             Optimizer.step()
#
#             # Validation step
#             if ModelType == "Sup":
#                 val_result = {"val_loss": LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)}
#             else:
#                 val_result = model.validation_step(Batch)
#
#             latest_loss = val_result["val_loss"]
#
#
#             # Save checkpoint
#             if PerEpochCounter % SaveCheckPoint == 0:
#                 SaveName = CheckPointPath + str(Epochs) + "a" + str(PerEpochCounter) + "model.ckpt"
#                 torch.save({
#                     "epoch": Epochs,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": Optimizer.state_dict(),
#                     "loss": latest_loss
#                 }, SaveName)
#                 print("\n" + SaveName + " Model Saved...")
#
#             # Tensorboard logging
#             Writer.add_scalar(
#                 "LossEveryIter",
#                 latest_loss,
#                 Epochs * NumIterationsPerEpoch + PerEpochCounter
#             )
#             Writer.flush()
#
#             # Explicit cleanup
#             if 'PredicatedCoordinatesBatch' in locals():
#                 del PredicatedCoordinatesBatch
#             del I1Batch, I2Batch, CoordinatesBatch, LossThisBatch
#             gc.collect()
#
#         # Save model after each epoch
#         SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
#         torch.save({
#             "epoch": Epochs,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": Optimizer.state_dict(),
#             "loss": latest_loss
#         }, SaveName)
#         print("\n" + SaveName + " Model Saved...")


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
        JsonPath
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
    # model = HomographyModel().to(device)
    # if ModelType == 'Sup': continue
    #     #model = HomographyModel().to(device)
    # elif ModelType == 'Unsup':
    model = UnsupervisedHomographyModel().to(device)

    result = None
    latest_loss = None
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        print("Epoch " + str(Epochs))
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Generate batch
            if ModelType == "Sup":
                I1Batch, I2Batch, CoordinatesBatch = GenerateBatch(JsonPath, MiniBatchSize)
            else:
                I1Batch, CornersBatch = UnSupGenerateBatch(JsonPath, MiniBatchSize, ModelType)
                I1Batch, CornersBatch = I1Batch.to(device), CornersBatch.to(device)

            # Predict output with forward pass
            #Batch = 0, I1Batch, I2Batch, CoordinatesBatch, 0

            # Training step
            if ModelType == "Sup":
                PredicatedCoordinatesBatch = model(I1Batch, I2Batch)
             #   LossThisBatch = LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)
                result = {"loss": LossThisBatch, "val_loss": LossThisBatch}
            else:
                Batch = (None, I1Batch[:, :3], I1Batch[:, 3:], CornersBatch, None)  # Ensure correct shape
                result = model.training_step(Batch, PerEpochCounter)
                LossThisBatch = result["loss"]


            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Validation step
            if ModelType == "Sup":
                val_result = {"val_loss": LossFn(PredicatedCoordinatesBatch, 0, 0, CoordinatesBatch)}
            else:
                val_result = model.validation_step(Batch)

            latest_loss = val_result["val_loss"]

            # Tensorboard logging
            Writer.add_scalar(
                "LossEveryIter",
                LossThisBatch,
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.add_scalar(
                "ValLossEveryIter",
                latest_loss,
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.flush()

            # Save checkpoint
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
                        "loss": latest_loss,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            # Explicitly delete tensors to free memory
            del I1Batch, I2Batch, CoordinatesBatch, LossThisBatch
            gc.collect()
            torch.mps.empty_cache()  # Optional: Clears GPU cache to free up memory

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": latest_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def plot_training_metrics(logs_path):
    """
    Plot training and validation loss from TensorBoard logs.
    Inputs:
        logs_path: Path to the TensorBoard logs.
    """
    # Read TensorBoard logs
    train_loss = []
    val_loss = []
    for event in SummaryWriter(logs_path).all_events():
        if event.tag == "LossEveryIter":
            train_loss.append(event.value)
        elif event.tag == "ValLossEveryIter":
            val_loss.append(event.value)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

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
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--JsonPath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Data\\train_json.json",
        help=f"JSON File Path, Default: /home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data/Val_json.json"
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="D:\\WPI Assignments\\Computer Vision CS549\\YourDirectoryID_p1\\YourDirectoryID_p1\\Phase2\\Checkpoints\\",
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
        default=24,
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
    JsonPath = Args.JsonPath

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
        JsonPath
    )

    # Plot training and validation metrics
    plot_training_metrics(LogsPath)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1)
