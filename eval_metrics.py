
import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc, jaccard_score

from matplotlib import pyplot as plt


print('Loading data')
os.chdir(r'C:\Users\researchlabs\Documents\spencers_stuff')

test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')

predictions = np.load('./data/predictions.npy')
print('Loading data complete')


# Building an ROC curve to evaluate model
a = predictions.flatten()
##a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

# sig_pred = tf.sigmoid(predictions)

thsh_index = np.argmax(tpr - fpr)
# print('Optinum threshold:', thresholds[thsh_index])

print('AUC:', roc_auc)

thshs = np.linspace(0, 1, 100)
thsh_opt = 0
iou_opt = 0

t = 0.02
binary = (a >= t).astype(int)
iou = jaccard_score(b, binary)
if iou > iou_opt:
    thsh_opt = t
    iou_opt = iou

print('Threshold:', thsh_opt)
print('IoU:', iou_opt)


# Visualizing
fig, ax = plt.subplots(4, 4, figsize=((8, 8)))

n = 800
k = np.arange(n, n+4)
for i in range(4):
    ax[i, 0].imshow(test_data[k[i], :, :, :])
    ax[i, 1].imshow(test_masks[k[i], :, :], cmap='gray')
    ax[i, 2].imshow(predictions[k[i], :, :, 0], cmap='plasma')
    # ax[i, 3].imshow(predictions[k[i], :, :, 0] > thresholds[thsh_index], cmap='gray')
    ax[i, 3].imshow(predictions[k[i], :, :, 0] > 0.02, cmap='gray')

for a in ax.flatten():
    a.axis('off')
plt.show()


