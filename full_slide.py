
'''
- break slide down into patches <- can use same alg! Minus sil.
- Perform same normalization on each.
- Obviously break down into GPU processing
- Register back to normal - using registration alg.
- Display - using same things.

* Should probably calculate in rows/batches... Otherwise will get out of control
    However, I'm going to try without this first...

If this does NOT work - I should try incoorporating more ZERO slides back in.

TODO: do a full pipelines, include data augmentation

'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.image import resize

import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex, AUROC
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ResnetUnet import SegmentationDataset, unet_dropout

from time import time


''' Global variables '''

target = 'H1_8-2-12-13'
slide_path = 'H1-2 Linear Reduced.tif'

patch_size = 112 # Can possibly bring this up to 224 if I generalize the model
patch_size_upsampled = 224
stride = 2 # Stride decreased from training. 4x overlap fine...

base_dir = r'C:\Users\researchlabs\Documents\Brown\Maiasaura'
data_dir = r'D:\Check Lab\Projects\OSU-CHS Projects\2018-05 Maiasaura Humeri\New Project Code Images'



'''
Functions from preprocessing
TODO: move this to a utils file
'''

def get_patch(patch):
    r, g, b = cv.split(patch)
    patch = np.stack([r, g, b], axis=2)
    patch = resize(patch, (patch_size_upsampled, patch_size_upsampled), method='bilinear')
    return patch
                

def norm_resnet(data):
    r = data[:, :, :, 0]
    g = data[:, :, :, 1]
    b = data[:, :, :, 2]

    r = (r - 123.68) / 58.393
    g = (g - 116.779) / 57.12
    b = (b - 103.939) / 57.375

    data = patch = np.stack([r, g, b], axis=3)
    return data
    

def from_subject_full_slide(subject, slide_path):
    patches = []
    patches_locs = []

    slide = cv.imread(f'./{subject}/{slide_path}')
    slide = cv.cvtColor(slide, cv.COLOR_BGR2RGB) # instead - convert to RGB for ResNet compatable.

    regx0 = 0
    regx1 = slide.shape[0]
    regy0 = 0
    regy1 = slide.shape[1]

    # Get number of patches in either dimension
    resx = np.floor((regx1 - regx0) / patch_size * stride).astype(int)
    resy = np.floor((regy1 - regy0) / patch_size * stride).astype(int)
    print('Expected number of patches:', resx*resy)

    patches = []
    masks = []
    masks_control = []

    for i in range(resx):
        for j in range(resy):
            a = regx0 + i * patch_size // stride
            b = regy0 + j * patch_size // stride

            h = slide[a:a + patch_size, b:b + patch_size]
            patches.append(h)

            patch_loc = np.array([a, b])
            patches_locs.append(patch_loc)

    patches_locs = np.stack(patches_locs, axis=0)

    patches = [get_patch(patches[i]) for i in range(len(patches))]
    patches_raw = np.stack(patches, axis=0)
    patches = np.stack(patches, axis=0)
    patches = norm_resnet(patches)

    # del sec, sil, slide, patches, masks  # Cleaning memory

    return slide, patches, patches_raw, patches_locs, resx, resy


''' Executing patch finder '''
t0 = time()
os.chdir(data_dir)
slide, patches, patches_raw, locs, resx, resy = from_subject_full_slide(target, slide_path)

print('Time to patch loading:', time() - t0)
os.chdir(base_dir)
np.save('H1_raw.npy', slide)
np.save('H1_patches.npy', patches)
np.save('H1_patches_raw.npy', patches_raw)
np.save('H1_locs.npy', locs)

# Takes 236 seconds to run above - simplifying now.
os.chdir(base_dir)
slide = np.load('H1_raw.npy')
patches = np.load('H1_patches.npy')
locs = np.load('H1_locs.npy')


''' Registration '''
# Importing model:
model_name = 'model_20250211_175346.pth'
model_path = f'C://Users/researchlabs/Documents/Brown/Maiasaura/models/{model_name}'
model = unet_dropout(backbone_name = 'resnet50',
                     encoder_weights='imagenet',
                     in_channels=3,
                     classes=1)


device = torch.device('cuda')
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()
torch.set_grad_enabled(False)

input_data = np.transpose(patches, (0, 3, 1, 2)) # input data

x_min = 0
x_max = int(np.max(locs[:, 0]) + patch_size)
y_min = 0
y_max = int(np.max(locs[:, 1]) + patch_size)

slide_pred = np.zeros(((x_max - x_min), (y_max - y_min)))
overlaps = np.zeros(((x_max - x_min), (y_max - y_min)))
print(locs.shape)
print(input_data.shape)

patches_pred = []

for i in range(locs.shape[0]):
    data = input_data[i, ...]
    torch_data = torch.from_numpy(data).unsqueeze(0).to(device)
    logits = model(torch_data).cpu()
    probs = torch.sigmoid(logits).numpy()[0, 0, :, :]

    data = np.transpose(data, (1, 2, 0))

    # Downsample from 244x244 -> 112x112:
    data = data[::2, ::2]
    patch = probs[::2, ::2]
    patches_pred.append(patch)
    
    a = locs[i, 0] - x_min
    b = locs[i, 1] - y_min

    slide_pred[a:a+patch_size, b:b+patch_size] = slide_pred[a:a+patch_size, b:b+patch_size] + patch
    overlaps[a:a+patch_size, b:b+patch_size] = overlaps[a:a+patch_size, b:b+patch_size] + np.ones((patch_size, patch_size))
    
patches_pred = np.stack(patches_pred, axis=0)

slide_pred[overlaps > 0] = slide_pred[overlaps > 0] / overlaps[overlaps > 0] # Finds mean

print('Total time elapsed:', time()- t0)
print('Saving')
os.chdir(base_dir)
np.save('H1_pred.npy', slide_pred)
np.save('H1_patches_pred.npy', patches_pred)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(slide)
ax[1].imshow(slide_pred, cmap='plasma')


for a in ax:
    a.axis('off')
plt.show()

