"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import json

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    try:
        DirNamesTrain = ['val__json']
        train_json_path = os.path.join(BasePath, 'val__json.json')
        if not os.path.exists(train_json_path):
            raise FileNotFoundError(f"JSON file not found at {train_json_path}")

        with open(train_json_path, 'r') as f:
            train_data = json.load(f)

        SaveCheckPoint = 100
        ImageSize = [128, 128, 6]  # Changed to match network expectations
        NumTrainSamples = len(train_data)
        TrainLabels = None
        NumClasses = None

        if not os.path.isdir(CheckPointPath):
            os.makedirs(CheckPointPath)

        return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses

    except Exception as e:
        print(f"Error in SetupAll: {str(e)}")
        sys.exit(1)

def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, "r")
        TrainLabels = TrainLabels.read()
        TrainLabels = list(map(float, TrainLabels.split()))

    return TrainLabels


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames("./TxtFiles/DirNamesTrain.txt")

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
