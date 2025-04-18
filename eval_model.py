
import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt


os.chdir(r'C:\Users\researchlabs\Documents\Brown\Maiasaura')
# Input parameters


test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')


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


model_name = './models/model_20250205_010758.keras'  # Model file name
model = models.load_model(model_name)

# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = 5, verbose=2)

np.save('./data/predictions.npy', predictions)


# Building an ROC curve to evaluate model
a = predictions.flatten()
##a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

# sig_pred = tf.sigmoid(predictions)

print('AUC:', roc_auc)

##plt.scatter(fpr, tpr, c=thresholds)
##plt.colorbar()
##plt.show()


#Optimizing:
thsh_index = np.argmax(tpr-fpr)

print('Optimum threshold:', thresholds[thsh_index])
print('Optimum sensitivity:', tpr[thsh_index])
print('Optimum specificity:', 1-fpr[thsh_index])


# Visualizing

for testy in range(predictions.shape[0]//40):
    fig, ax = plt.subplots(4, 4, figsize=((8, 8)))
    
    n = testy * 40
    k = np.arange(n, n+4)
    for i in range(4):
        ax[i, 0].imshow(test_data[k[i], :, :, :])
        ax[i, 1].imshow(test_masks[k[i], :, :], cmap='gray')
        ax[i, 2].imshow(predictions[k[i], :, :, 0], cmap='plasma')
        ax[i, 3].imshow(predictions[k[i], :, :, 0] > thresholds[thsh_index], cmap='gray')
        # ax[i, 3].imshow(predictions[k[i], :, :, 0] > -1, cmap='gray')

    for a in ax.flatten():
        a.axis('off')
    plt.show()
