import numpy as np
import cv2
import random

from albumentations import (Resize, Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, RandomSizedCrop, RandomCrop,
                            RandomBrightnessContrast, HueSaturationValue, RandomGamma, RandomBrightnessContrast,
                            GaussianBlur, ColorJitter, Emboss, GaussNoise, ChannelShuffle, Normalize, OneOf)
from albumentations.augmentations.utils import (is_rgb_image, is_grayscale_image)
from albumentations.core.transforms_interface import (ImageOnlyTransform)
from albumentations.pytorch import ToTensorV2


class Affine(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Affine, self).__init__(always_apply, p)
    
    def apply(self, img, **params):
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("RandomAffine transformation expects 1-channel or 3-channel images.")
        matrix1 = self.buildTransformationMatrix()
        matrix2 = self.buildTransformationMatrix()
        M = cv2.getAffineTransform(matrix1, matrix2)
        cols, rows, _ = img.shape
        return cv2.warpAffine(img, M, (cols, rows))
        
    def buildTransformationMatrix(self):
        """ Standard range for affine transformations on this matrix """
        return np.float32([
            [random.randrange(40, 60), random.randrange(40, 60)],
            [random.randrange(190, 210), random.randrange(40, 60)],
            [random.randrange(90, 110), random.randrange(90, 110)],
        ])
    


def get_transforms(width, height):
    transforms = Compose([
        Resize(width=width, height=height),
        # Composition
        RandomCrop(width=width, height=height, p=1),
        RandomRotate90(p=1),
        HorizontalFlip(p=0.3),
        VerticalFlip(p=0.3),
        Affine(p=0.5),
        
        # Image noise
        GaussianBlur(blur_limit=(1, 3), p=0.3),
        
        # Colorize
        RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.3, 
                                 brightness_by_max=True,p=0.5),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, 
                           val_shift_limit=0, p=0.5),
        
        OneOf([
            ChannelShuffle(p=1),
            ColorJitter(p=1),
        ]),
    ])
    
    return transforms

def normalize():
    """
    Normalize image 
    """
    norm = Compose([
        # Finalize
        Normalize(
            mean=[0,0,0],
            std=[1,1,1],
            max_pixel_value=255
        ),
    ])
    return norm