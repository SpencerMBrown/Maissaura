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
epochs = 25
n = 224

train_data = np.load('./data/train_data.npy')
train_masks = np.load('./data/train_masks.npy')
test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')
val_data = np.load('./data/val_data.npy')
val_masks = np.load('./data/val_masks.npy')

print('Completed data inititialization')


pos = np.sum(train_masks) / np.prod(train_masks.shape)
pos_weight = (1 - pos) / pos
print('Weight:', pos_weight)


##@keras.saving.register_keras_serializable()
##def weighted_loss_with_logits(y_true, y_pred):
##    loss = tf.nn.weighted_cross_entropy_with_logits(
##        labels=y_true, logits=y_pred, pos_weight=pos_weight
##    )
##    return tf.reduce_mean(loss)

model_name = './models/model_20250127_033826.keras'  # Model file name
model = models.load_model(model_name)

i = 0
# Now - resetting all layers to true... Trying to tune it down. 
for layer in model.layers:
    layer.trainable = True

# Recompiling:
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

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
# a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)



# Saving model:
model_name_new = f'{model_name.split('.keras')[0]}_retrain.keras'
model.save(model_name_new)
print(f'Model saved as {model_name_new}')

thshs = np.linspace(0, 1, 100)
thsh_opt = 0
iou_opt = 0

for t in thshs:
    binary = (a >= t).astype(int)
    iou = jaccard_score(b, binary)
    if iou > iou_opt:
        thsh_opt = t
        iou_opt = iou

print('Threshold:', thsh_opt)
print('IoU:', iou_opt)
