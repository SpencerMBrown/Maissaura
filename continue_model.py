import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc

from datetime import datetime


os.chdir(r'C:\Users\researchlabs\Documents\spencers_stuff')
# Input parameters

batch_size = 5
epochs = 500
n = 128

train_data = np.load('./models/train_data.npy')
train_masks = np.load('./models/train_masks.npy')
test_data = np.load('./models/test_data.npy')
test_masks = np.load('./models/test_masks.npy')
val_data = np.load('./models/val_data.npy')
val_masks = np.load('./models/val_masks.npy')
masks = np.load('./models/masks.npy')


pos = np.sum(masks) / np.prod(masks.shape)
pos_weight = (1 - pos) / pos


@keras.saving.register_keras_serializable()
def weighted_loss_with_logits(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    return tf.reduce_mean(loss)

model_name = './models/model_20250110_112335.keras'  # Model file name
model = models.load_model(model_name)


##########################################################
# Coninue training for previous model
##########################################################
history = model.fit(train_data, train_masks,
              validation_data=(val_data, val_masks),
              batch_size=batch_size,
              epochs=epochs,
              verbose=2)


# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = batch_size)


# Building an ROC curve to evaluate model
a = predictions.flatten()
a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)


# Saving model:
model.save(model_name)
print(f'Model saved as {model_name}')
