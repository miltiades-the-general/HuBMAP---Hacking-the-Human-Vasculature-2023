import pandas as pd
import numpy as np
import random
import json
import os
from collections import defaultdict

import tifffile as tif
import cv2
import matplotlib.pyplot as plt
from IPython.display import IFrame
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torchvision.transforms as T

import tensorflow as tf

from transforms.transforms import normalize, get_transforms

class hubmapDataPoint():
    def __init__(self, image_id, image, mask):
        self.id = image_id
        self.image = image
        self.mask = mask
    
    def getImage(self):
        """
        Returns image array
        """
        return self.image
    
    def getMask(self):
        """
        Returns the mask array
        """
        return self.mask
    
    def getId(self):
        """
        Returns the images unique id
        """
        return self.id

class HuBMAPDataset(Dataset):
    def __init__(self, train_image_dir=CFG.train_image_dir, labels_dir=CFG.labels_dir, augments=CFG.augments, mask_color="black"):
        self.train_image_dir = train_image_dir
        self.augments = augments
        self.mask_color = { 'white': [1,1,1], 'red': [1,0,0], 'green': [0,1,0], 'blue': [0,0,1], 'black': [-1,-1,-1]}
        self.mask_color = self.mask_color[mask_color]
        
        with open(labels_dir, 'r') as json_file:
            self.json_labels = [json.loads(line) for line in json_file]
    
    __len__ = lambda self : len(self.json_labels)
    
    def __getitem__(self, idx):
        image_id = self.json_labels[idx]['id']
        image_path = self.train_image_dir + f"/{image_id}.tif"
        image = tif.imread(image_path)
        
        label = self.json_labels[idx]
        mask = np.zeros((CFG.HEIGHT, CFG.WIDTH, 3), dtype=np.float32)
        
        # Finalize
        norm = normalize(CFG.WIDTH, CFG.HEIGHT)
        image = norm(image=image)["image"]
        
        # Each label contains a number of maskings
        for annotation in label['annotations']:
            coords = annotation['coordinates']
            coordinates = defaultdict(list)

            # The target is blood vessels
            if annotation['type'] == 'blood_vessel':
                for coord in coords:
                    row_coords, col_coords = np.array([i[1] for i in coord]), np.array([i[0] for i in coord])
                    coordinates = defaultdict(list)
                    for row, col in zip(row_coords, col_coords):
                        coordinates[row].append(col)

                for row, cols in coordinates.items():
                    start = min(cols)
                    end = max(cols)

                    for col in range(start, end + 1):
                        mask[row, col] = self.mask_color
                        
        # Apply the mask to the image
        image = image + mask # C, H, W
        
        if self.augments:
            transforms = get_transforms(CFG.WIDTH, CFG.HEIGHT)
            image = transforms(image=image)["image"]
            
        image = torch.tensor(np.array(image), dtype=torch.float32) 

        mask = torch.tensor(mask, dtype=torch.float32)
        
        datapoint = hubmapDataPoint(image_id, image, mask)
        return datapoint