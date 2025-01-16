import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras import layers, models

from sklearn.metrics import roc_curve, auc

from datetime import datetime


#os.chdir(r'C:\Users\researchlabs\Documents\spencers_stuff')
os.chdir('/')
# Input parameters

batch_size = 5
epochs = 5
n = 96

train_data = np.load('./models/train_data.npy')
train_masks = np.load('./models/train_masks.npy')
test_data = np.load('./models/test_data.npy')
test_masks = np.load('./models/test_masks.npy')
val_data = np.load('./models/val_data.npy')
val_masks = np.load('./models/val_masks.npy')
masks = np.load('./models/masks.npy')



# U-NET architecture:
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    outputs = layers.Conv2D(1, (1, 1))(c9)


    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model



pos = np.sum(masks) / np.prod(masks.shape)
print(f'Pos rate: {pos}')
pos_weight = (1 - pos) / pos
pos_weight = pos_weight * 0.8 # Smoothing to try to stop model collapse.


@keras.saving.register_keras_serializable()
def weighted_loss_with_logits(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    return tf.reduce_mean(loss)

model = build_unet(input_shape=(n, n, 6))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',
              loss=weighted_loss_with_logits,
              metrics=['accuracy'])



##########################################################
# Model initial training
##########################################################
history = model.fit(train_data, train_masks,
              validation_data=(val_data, val_masks),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)


# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = batch_size,
                            verbose = 2)


# Building an ROC curve to evaluate model
a = predictions.flatten()
a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)


# Saving model:
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
model_name = f'./models/model_{current_time}.keras'  # Model file name
model.save(model_name)
print(f'Model saved as {model_name}')
