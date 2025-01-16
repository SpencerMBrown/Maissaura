
import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt


os.chdir('/Users/spencerbrown/PycharmProjects/Maiasaura/OneDrive_1_1-10-2025')
# Input parameters


test_data = np.load('test_data.npy')
test_masks = np.load('test_masks.npy')

#
# pos = np.sum(masks) / np.prod(masks.shape)
# pos_weight = (1 - pos) / pos
pos_weight = 3 # For tests.

@keras.saving.register_keras_serializable()
def weighted_loss_with_logits(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    return tf.reduce_mean(loss)

model_name = 'model_20250110_112335 - 100epoch.keras'  # Model file name
model = models.load_model(model_name)


# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = 5)


# Building an ROC curve to evaluate model
a = predictions.flatten()
a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)


#Optimizing:
thsh_index = np.argmax(tpr-fpr)

print('Optimum threshold:', thresholds[thsh_index])
print('Optimum sensitivity:', tpr[thsh_index])
print('Optimum specificity:', 1-fpr[thsh_index])


# Visualizing

k = 0

fig, ax = plt.subplots(2, 2, figsize=((8, 8)))

ax[0, 0].imshow(test_data[k, :, :, 1], cmap='plasma')
ax[1, 0].imshow(test_masks[k, :, :], cmap='gray')
ax[0, 1].imshow(predictions[k, :, :, 0], cmap='plasma')
ax[1, 1].imshow(predictions[k, :, :, 0] > thresholds[thsh_index], cmap='gray')

for a in ax.flatten():
    a.axis('off')
plt.show()