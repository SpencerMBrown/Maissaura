import numpy as np

import torch
import torch.nn as nn
from torchmetrics.classification import JaccardIndex, AUROC
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ResnetUnet import SegmentationDataset, unet_dropout
from sklearn.metrics import roc_curve, auc, jaccard_score

from matplotlib import pyplot as plt

patch_size=112
batch_size = 32

test_locs = np.load('./data/test_locs.npy')


''' Importing model: '''
model_name = 'model_20250211_175346.pth'
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



x_min = np.min(test_locs[:, 0])
x_max = np.max(test_locs[:, 0]) + patch_size
y_min = np.min(test_locs[:, 1])
y_max = np.max(test_locs[:, 1]) + patch_size

x_min = int(x_min)
x_max = int(x_max)
y_min = int(y_min)
y_max = int(y_max)

slide = np.zeros(((x_max - x_min), (y_max - y_min)))
masks = np.zeros(((x_max - x_min), (y_max - y_min)))
overlaps = np.zeros(((x_max - x_min), (y_max - y_min)))
    # Counts number of overlaps for each pixel.

print(test_locs.shape)
print(test_data.shape)



for i in range(test_locs.shape[0]):
    data = test_data[i, ...]
    logits = model(torch.from_numpy(data).unsqueeze(0))
    probs = torch.sigmoid(logits).numpy()[0, 0, :, :]

    data = np.transpose(data, (1, 2, 0))
    mask = np.transpose(test_masks[i, :, :], (1, 2, 0))

    # Downsample:
    data = data[::2, ::2]
    mask = mask[::2, ::2][:, :, 0]
    patch = probs[::2, ::2]

    a = test_locs[i, 0] - x_min
    b = test_locs[i, 1] - y_min

    slide[a:a+patch_size, b:b+patch_size] = slide[a:a+patch_size, b:b+patch_size] + patch
    masks[a:a+patch_size, b:b+patch_size] = masks[a:a+patch_size, b:b+patch_size] + mask
    overlaps[a:a+patch_size, b:b+patch_size] = overlaps[a:a+patch_size, b:b+patch_size] + np.ones((patch_size, patch_size))
    
##    for j in range(patch_size):
##        slide[a:a+patch_size, (b+j)] = slide[a:a+patch_size, (b+j)] + patch[:, j]
##        masks[a:a+patch_size, (b+j)] = masks[a:a+patch_size, (b+j)] + mask[:, j]
##        overlaps[a:a+patch_size, (b+j)] = overlaps[a:a+patch_size, (b+j)] + np.ones(patch_size)

slide[overlaps > 0] = slide[overlaps > 0] / overlaps[overlaps > 0] # Finds mean
masks = masks > 0


# Building an ROC curve to evaluate model in it's entirety
pred = slide.flatten()
truth = masks.flatten()
c = overlaps.flatten()

print('Original shape:', len(truth))
pred = pred[c > 0]
truth = truth[c > 0]
print('Filtered shape:', len(pred))

fpr, tpr, thresholds = roc_curve(truth, pred)
roc_auc = auc(fpr, tpr)


thsh_index = np.argmax(tpr-fpr)
print('AUC:', roc_auc)
print('Optimum threshold:', thresholds[thsh_index])
print('Optimum sensitivity:', tpr[thsh_index])
print('Optimum specificity:', 1-fpr[thsh_index])

pred_binary = pred >= thresholds[thsh_index]
print('IoU:', jaccard_score(truth, pred_binary))

plt.scatter(fpr, tpr, c=thresholds)
plt.colorbar()
plt.show()



fig, ax = plt.subplots(1, 3, figsize=(8, 8))
ax[0].imshow(slide)
ax[1].imshow(overlaps)
ax[2].imshow(masks, cmap='gray')

for a in ax:
    a.axis('off')
plt.show()
