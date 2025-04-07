# TODO:
    # Set up some type of early stopping
    # Fizx processing - fully separate val dataset, but within H40 or...
    # Consdier swapping tst dataset around.


import numpy as np
import os
import sys
import io
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp
from tqdm import tqdm

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True) # or 'forkserver'

from sklearn.metrics import roc_auc_score
from torcheval.metrics.aggregation.auc import AUC


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
model_name = f'./models/model_{current_time}.pth'  # Model file name

os.chdir('C://Users/researchlabs/Documents/Brown/Maiasaura')
# Input parameters

batch_size = 10
epochs = 100 # Maximum epochs without callback
n = 224


'''
Model definition
'''


class unet_dropout(smp.Unet):
    def __init__(self, dropout_rate = 0.5, *args, **kwargs):
        super(unet_dropout, self).__init__(*args, **kwargs)
    

        for i, block in enumerate(self.decoder.blocks):
            if isinstance(block, nn.Sequential):
                block.add_module(f'dropout_{i}', nn.Dropout2d(p=dropout_rate))

    def forward(self, x):
        return super(unet_dropout, self).forward(x)


''' Dataset class '''

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask


