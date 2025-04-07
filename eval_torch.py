import numpy as np

import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex, AUROC
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ResnetUnet import SegmentationDataset, unet_dropout

from matplotlib import pyplot as plt

batch_size = 32


''' Importing model: '''
model_name = 'model_20250208_204548.pth'
model_path = f'C://Users/researchlabs/Documents/Brown/Maiasaura/models/{model_name}'
model = unet_dropout(backbone_name = 'resnet50',
                     encoder_weights='imagenet',
                     in_channels=3,
                     classes=1)

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
torch.set_grad_enabled(False)


''' Dataset '''
test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')
test_data = np.transpose(test_data, (0, 3, 1, 2))
test_masks = test_masks[:, np.newaxis, :, :]
test_dataset = SegmentationDataset(test_data, test_masks)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


''' Calculate metrics '''
y_score = []
y_truth = []

for x, y in test_loader:
    logits = model(x).squeeze()
    probs = torch.sigmoid(logits)
    y_score.append(probs)
    y_truth.append(y)

y_score = torch.cat(y_score).flatten()
y_truth = torch.cat(y_truth).flatten().to(torch.int)

auc_metric = AUROC(task='binary')
iou_metric = JaccardIndex(task='binary')

auc = auc_metric(y_score, y_truth)
iou = iou_metric(y_score, y_truth)

print(f'AUC: {auc:.4f}')
print(f'IoU: {iou:.4f}')

# TODO: add some type of visualization here.

print(test_data.shape)

''' Visualizing '''
testy = np.random.choice(np.arange(test_data.shape[0]), 10)
for i in testy:
    data = test_data[i, ...]
    logits = model(torch.from_numpy(data).unsqueeze(0))
    probs = torch.sigmoid(logits).numpy()[0, 0, :, :]

    data = np.transpose(data, (1, 2, 0))
    mask = np.transpose(test_masks[i, :, :], (1, 2, 0))

    print(data.shape)
    
    fig, ax = plt.subplots(1, 4, figsize=((12, 3)))

    ax[0].imshow(data)
    ax[1].imshow(mask, cmap='gray')
    ax[2].imshow(probs, cmap='plasma')
    ax[3].imshow(probs > 0.5, cmap='gray')

    for a in ax.flatten():
        a.axis('off')
    plt.show()


