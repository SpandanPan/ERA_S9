from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np

train_transforms = A.Compose([
    #A.Resize(32, 32),
    A.ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.1, rotate_limit = 15,p=0.4),
    A.HorizontalFlip(),
    A.ColorJitter (brightness=0.1, contrast=0.2, saturation=0.4, hue=0.2, always_apply=False, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, mask_fill_value=None, always_apply=False, p=0.3),
    A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ToTensorV2()
])

# Test Phase transformations
test_transforms = A.Compose([
                             ToTensorV2(),
                             A.Normalize (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                                       ])

test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

def apply_transform_train(image):
    return train_transforms(image=np.array(image))

def apply_transform_test(image):
    return test_transforms(image=np.array(image))

