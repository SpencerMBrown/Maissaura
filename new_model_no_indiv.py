import numpy as np
import os

import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from keras.callbacks import CSVLogger, EarlyStopping

from sklearn.metrics import roc_curve, auc

from datetime import datetime


os.chdir('C://Users/researchlabs/Documents/Brown/Maiasaura')
# Input parameters

batch_size = 10
epochs = 50 # Maximum epochs without callback
n = 224

train_data = np.load('./data/train_data.npy')
train_masks = np.load('./data/train_masks.npy')
test_data = np.load('./data/test_data.npy')
test_masks = np.load('./data/test_masks.npy')
val_data = np.load('./data/val_data.npy')
val_masks = np.load('./data/val_masks.npy')


# New, smaller U-NET architecture:
##dor = 0.2
##def build_model(input_shape):
##    inputs = layers.Input(shape=input_shape)
##
##    # Contracting Path (Encoder)
##    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
##    x = layers.Dropout(dor)(x)
##    
##    x = layers.MaxPooling2D((2, 2))(x)
##    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
##    x = layers.Dropout(dor)(x)
##    
##    x = layers.MaxPooling2D((2, 2))(x)
##    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
##    x = layers.Dropout(dor)(x)
##
##    x = layers.MaxPooling2D((2, 2))(x)
##    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
##    x = layers.Dropout(dor)(x)
##    
##    x = layers.MaxPooling2D((2, 2))(x)
##
##    # Expansive Path (Decoder)
##    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(x)
##    x = layers.Concatenate()([x, layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)])
##    x = layers.Dropout(dor)(x)
##    
##    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(x)
##    x = layers.Concatenate()([x, layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)])
##    x = layers.Dropout(dor)(x)
##
##    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu')(x)
##    x = layers.Concatenate()([x, layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)])
##    x = layers.Dropout(dor)(x)
##    
##    # Final Convolutional Layer
##    outputs = layers.Conv2D(1, (1, 1))(x)
##
##    # Define the model
##    model = models.Model(inputs=inputs, outputs=outputs)
##
##    return model


###########################################################################
# UNET++ architecture:
###########################################################################

### Defining the Convolutional Block
##dor = 0.2 # Drop out rate
##
##def conv_block(inputs, num_filters):
##    # Applying the sequence of Convolutional, Batch Normalization
##    # and Activation Layers to the input tensor
##    x = tf.keras.Sequential([
##        # Convolutional Layer
##        layers.Conv2D(num_filters, 3, padding='same'),
##        # Batch Normalization Layer
##        layers.BatchNormalization(),
##        # Activation Layer
##        layers.Activation('relu'),
##        # Convolutional Layer
##        layers.Conv2D(num_filters, 3, padding='same'),
##        # Batch Normalization Layer
##        layers.BatchNormalization(),
##        # Activation Layer
##        layers.Activation('relu'),
##        # Dropout
##        layers.Dropout(dor)
##    ])(inputs)
## 
##    # Returning the output of the Convolutional Block
##    return x
##
##def build_model(input_shape):
##    inputs = layers.Input(shape=input_shape)
#### 
##     # Encoding Path
##    x_00 = conv_block(inputs, 64)
##    x_10 = conv_block(tf.keras.layers.MaxPooling2D()(x_00), 128)
##    x_20 = conv_block(tf.keras.layers.MaxPooling2D()(x_10), 256)
##    x_30 = conv_block(tf.keras.layers.MaxPooling2D()(x_20), 512)
##    x_40 = conv_block(tf.keras.layers.MaxPooling2D()(x_30), 1024)
## 
##    # Nested Decoding Path
##    x_01 = conv_block(tf.keras.layers.concatenate(
##        [x_00, tf.keras.layers.UpSampling2D()(x_10)]), 64)
##    x_11 = conv_block(tf.keras.layers.concatenate(
##        [x_10, tf.keras.layers.UpSampling2D()(x_20)]), 128)
##    x_21 = conv_block(tf.keras.layers.concatenate(
##        [x_20, tf.keras.layers.UpSampling2D()(x_30)]), 256)
##    x_31 = conv_block(tf.keras.layers.concatenate(
##        [x_30, tf.keras.layers.UpSampling2D()(x_40)]), 512)
## 
##    x_02 = conv_block(tf.keras.layers.concatenate(
##        [x_00, x_01, tf.keras.layers.UpSampling2D()(x_11)]), 64)
##    x_12 = conv_block(tf.keras.layers.concatenate(
##        [x_10, x_11, tf.keras.layers.UpSampling2D()(x_21)]), 128)
##    x_22 = conv_block(tf.keras.layers.concatenate(
##        [x_20, x_21, tf.keras.layers.UpSampling2D()(x_31)]), 256)
## 
##    x_03 = conv_block(tf.keras.layers.concatenate(
##        [x_00, x_01, x_02, tf.keras.layers.UpSampling2D()(x_12)]), 64)
##    x_13 = conv_block(tf.keras.layers.concatenate(
##        [x_10, x_11, x_12, tf.keras.layers.UpSampling2D()(x_22)]), 128)
## 
##    x_04 = conv_block(tf.keras.layers.concatenate(
##        [x_00, x_01, x_02, x_03, tf.keras.layers.UpSampling2D()(x_13)]), 64)
##
##
##    outputs = layers.Conv2D(1, (1, 1))(x_02)
##    
##    # Creating the model
##    model = tf.keras.Model(
##        inputs=inputs, outputs=outputs, name='Unet_plus_plus')
## 
##    # Returning the model
##    return model



###########################################################################
# ResNet34 based encoder
###########################################################################

dor = 0.2 # Drop out rate

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(dor)(x)
 
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(dor)(x)
 
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_model(input_shape):
    """ Input """
    inputs = Input(input_shape)
 
    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
 
    """ Encoder and skip points """
    s1 = resnet50.get_layer("input_layer").output
    
    s2 = resnet50.get_layer("conv1_relu").output        
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
 
    """ Bridge """
    b = resnet50.get_layer("conv4_block6_out").output
 
    """ Decoder """
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)


    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation='sigmoid')(d4)
 
    model = Model(inputs, outputs, name="ResNet_UNet")
    return model

# Up next - add attention, switch skip connection origins. 



##pos = np.sum(train_masks) / np.prod(train_masks.shape)
##pos_weight = (1 - pos) / pos
##pos_weight = pos_weight * 1 # Smoothing to try to stop model collapse.
##print('Weight:', pos_weight) 


# Weighted binary cross-entropy loss function
##@keras.saving.register_keras_serializable()
##def weighted_loss_with_logits(y_true, y_pred):

##    y_true = tf.repeat(y_true, repeats=3, axis=0)
##    print(y_true.shape)
##    print(y_pred.shape)
##    
##        
##    loss = tf.nn.weighted_cross_entropy_with_logits(
##        labels=y_true, logits=y_pred, pos_weight=pos_weight
##    )
##    return tf.reduce_mean(loss)



# Building model
csv_logger = CSVLogger('./models/log.csv', append=True, separator=',')

model = build_model(input_shape=(n, n, 3))

# Freezing encoding for early layers
for layer in model.layers[:80]:
    layer.trainable = False


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss=weighted_loss_with_logits,
#               metrics=['accuracy'])

print(model.summary())


##########################################################
# Model initial training
##########################################################
callback = EarlyStopping(monitor='val_loss', patience=10,
                         restore_best_weights=True)

##history = model.fit(train_data, train_masks,
##                    validation_data=(val_data, val_masks),
##                    batch_size=batch_size,
##                    epochs=1,
##                    verbose=1,
##                    callbacks=[callback])

history = model.fit(train_data, train_masks,
                    validation_data=(val_data, val_masks),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[callback])


##########################################################
# Stage 2 model training
##########################################################

for layer in model.layers[40:]:
    layer.trainable = True

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_masks,
                    validation_data=(val_data, val_masks),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[callback])


##########################################################
# Model analysis
##########################################################

# Predicting from test dataset:
predictions = model.predict(test_data,
                            batch_size = batch_size,
                            verbose=2)


# Building an ROC curve to evaluate model
a = predictions.flatten()
# a = tf.sigmoid(a)
b = test_masks.flatten()

fpr, tpr, thresholds = roc_curve(b, a)
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)


# Saving model:
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
model_name = f'./models/model_{current_time}.keras'  # Model file name
model.save(model_name)
print(f'Model saved as {model_name}')

