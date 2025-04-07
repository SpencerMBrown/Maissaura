import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt


os.chdir(r'C:\Users\researchlabs\Documents\spencers_stuff')

# Input parameters
patch_size = 224-32
buffer = 16 # on either side.

test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')
test_locs = np.load('./data/test_locs.npy')

print('data loading complete')


#
# pos = np.sum(masks) / np.prod(masks.shape)
# pos_weight = (1 - pos) / pos
##pos_weight = 3 # For tests.
##
##@keras.saving.register_keras_serializable()
##def weighted_loss_with_logits(y_true, y_pred):
##    loss = tf.nn.weighted_cross_entropy_with_logits(
##        labels=y_true, logits=y_pred, pos_weight=pos_weight
##    )
##    return tf.reduce_mean(loss)


model_name = './models/model_20250127_033826.keras'  # Model file name
model = models.load_model(model_name)

print('model loading complete')

# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = 50, verbose=2)

# Testing what happens if I do a sigmnoid activation...
# predictions = tf.sigmoid(predictions)


# Code for registration -> a single prediction.
# For now - using mean prediction theorized to bring out ground truth
    # via law of large numbers.

x_min = np.min(test_locs[:, 0])
x_max = np.max(test_locs[:, 0]) + patch_size
y_min = np.min(test_locs[:, 1])
y_max = np.max(test_locs[:, 1]) + patch_size

slide = np.zeros(((x_max - x_min), (y_max - y_min)))
masks = np.zeros(((x_max - x_min), (y_max - y_min)))
overlaps = np.zeros(((x_max - x_min), (y_max - y_min)))
    # Counts number of overlaps for each pixel.

print(test_locs.shape)
print(predictions.shape)


for i in range(test_locs.shape[0]):

    patch = predictions[i, :, :, 0]
    patch = patch[buffer:-buffer, buffer:-buffer]

    mask = test_masks[i, :, :]
    mask = mask[buffer:-buffer, buffer:-buffer]

    a = test_locs[i, 0] - x_min
    b = test_locs[i, 1] - y_min

    for j in range(patch_size):
        slide[a:a+patch_size, (b+j)] = slide[a:a+patch_size, (b+j)] + patch[:, j]
        masks[a:a+patch_size, (b+j)] = masks[a:a+patch_size, (b+j)] + mask[:, j]
        overlaps[a:a+patch_size, (b+j)] = overlaps[a:a+patch_size, (b+j)] + np.ones(patch_size)

slide[overlaps > 0] = slide[overlaps > 0] / overlaps[overlaps > 0]
masks = masks > 0


# Building an ROC curve to evaluate model in it's entirety
a = slide.flatten()
b = masks.flatten()
c = overlaps.flatten()

print('Original shape:', len(a))
a = a[c > 0]
b = b[c > 0]
print('Filtered shape:', len(a))

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)


thsh_index = np.argmax(tpr-fpr)
print('AUC:', roc_auc)
print('Optimum threshold:', thresholds[thsh_index])
print('Optimum sensitivity:', tpr[thsh_index])
print('Optimum specificity:', 1-fpr[thsh_index])

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


