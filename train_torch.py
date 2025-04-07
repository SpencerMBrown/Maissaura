# TODO:
    # Set up some type of early stopping
    # Don't freeze layers so late.
    # Maybe add more data, see what happens?

    # And fainly - compare overall ranges.. and predict




import numpy as np
import os
import sys
import contextlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp
from tqdm import tqdm

import torch.multiprocessing as
mp
mp.set_start_method('spawn', force=True) # or 'forkserver'

from torchmetrics.classification import JaccardIndex, AUROC

from ResnetUnet import SegmentationDataset, unet_dropout


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
model_name = f'./models/model_{current_time}.pth'  # Model file name

os.chdir('C://Users/researchlabs/Documents/Brown/Maiasaura')
# Input parameters

batch_size = 10
epochs = 300 # Maximum epochs without callback
n = 224


'''
Model definition
'''

# Ridirect download output from torch for updated ResNet weights
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        model = unet_dropout(dropout_rate=0.5,
                             backbone_name = 'resnet50',
                             encoder_weights='imagenet',
                             in_channels=3,
                             classes=1)


device = torch.device('cuda')
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')


'''
Training cycles
'''

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False, delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_best_model()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stop')
                return True

        else:
            self.best_score = score
            self.save_best_model()
            self.counter= 0

        return False

    def save_best_model(self):
        self.best_model = model.state_dict()

def train_one_epoch(dataloader):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, disable=True)

    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        # loop.set_description(f'Loss: {loss.item():.4f}')

    return running_loss / len(dataloader)

def validate(dataloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(dataloader)


early_stopping = EarlyStopping()
def train_model(train_loader, val_loader):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        train_loss = train_one_epoch(train_loader)
        val_loss = validate(val_loader)

        print(f'Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)

        if early_stopping(val_loss):
            model.load_state_dict(early_stopping.best_model)
            torch.save(model.state_dict(), model_name) # Resave best model
            break


'''
Loading data, creating dataset
'''

train_data = np.load('./data/train_data.npy')
train_masks = np.load('./data/train_masks.npy')
val_data = np.load('./data/val_data.npy')
val_masks = np.load('./data/val_masks.npy')
test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')


train_data = np.transpose(train_data, (0, 3, 1, 2))
train_masks = train_masks[:, np.newaxis, :, :]
val_data = np.transpose(val_data, (0, 3, 1, 2))
val_masks = val_masks[:, np.newaxis, :, :]
test_data = np.transpose(test_data, (0, 3, 1, 2))
test_masks = test_masks[:, np.newaxis, :, :]


train_dataset = SegmentationDataset(train_data, train_masks)
val_dataset = SegmentationDataset(val_data, val_masks)
test_dataset = SegmentationDataset(test_data, test_masks)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


'''
Training the model
'''

# Freezing encoder layers - 108 total
i = 0
for param in model.encoder.parameters():
    if i < 108: # Cutoff - allow early ResNet layers
        param.requires_grad = False
    i += 1

train_model(train_loader, val_loader)

i = 0
for param in model.encoder.parameters():
    param.requires_grad = True
    if i < 40: # Cutoff - allow early ResNet layers
        param.requires_grad = False
    i += 1

train_model(train_loader, val_loader)

# Unfreezing all encoder layers
for param in model.encoder.parameters():
    param.requires_grad = True

train_model(train_loader, val_loader)


'''
Eval model
'''
model.eval()
torch.set_grad_enabled(False)

y_score = []
y_truth = []

for x, y in test_loader:
    x, y = x.cuda(), y.cuda()
    
    logits = model(x).squeeze()
    probs = torch.sigmoid(logits)
    y_score.append(probs)
    y_truth.append(y)

y_score = torch.cat(y_score).flatten()
y_truth = torch.cat(y_truth).flatten().to(torch.int)

auc_metric = AUROC(task='binary').to(device)
iou_metric = JaccardIndex(task='binary').to(device)

auc = auc_metric(y_score, y_truth)
iou = iou_metric(y_score, y_truth)

print(f'AUC: {auc:.4f}')
print(f'IoU: {iou:.4f}')
