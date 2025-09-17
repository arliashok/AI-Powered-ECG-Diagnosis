#!/usr/bin/env python
# coding: utf-8

"""### Import Libraries"""
from tensorflow.keras.layers import (Conv2D, Add, Activation,
                                     Dropout, Dense, Flatten, Input, BatchNormalization,
                                     ReLU, MaxPooling2D, Concatenate, GlobalAveragePooling2D
                                     )
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses, metrics, regularizers, callbacks
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import os


"""#### Import Data"""

# Use same path where x_train.npy, y_train.npy, etc. were saved in previous script
path = 'D:\\ECG\\'

calssificatin_type = {"binary": 1, "superclasses": 5, "subclasses": 23}
classification_name = "subclasses"
no_of_classes = calssificatin_type[classification_name]

lead_type = {"lead-I": 1, "bipolar-limb": 3, "unipolar-limb": 3, "limb-leads": 6,
             "precordial-leads": 6, "all-lead": 12}
lead_name = "all-lead"
no_of_leads = lead_type[lead_name]

# Load saved numpy arrays
x_train = np.load(path + 'x_train.npy', allow_pickle=True)
x_test = np.load(path + 'x_test.npy', allow_pickle=True)
y_train = np.load(path + 'y_train.npy', allow_pickle=True)
y_test = np.load(path + 'y_test.npy', allow_pickle=True)

# Reshape for CNN
x_train = x_train.transpose(0, 2, 1)   # transpose correctly
x_test = x_test.transpose(0, 2, 1)

x_train = x_train.reshape(x_train.shape[0], no_of_leads, 1000, 1)   # Add channel dim
x_test = x_test.reshape(x_test.shape[0], no_of_leads, 1000, 1)

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)
print('Data loaded')

from sklearn.preprocessing import MultiLabelBinarizer

if classification_name != "binary":
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)
    y_train = mlb.transform(y_train)

    mlb = MultiLabelBinarizer()
    mlb.fit(y_test)
    y_test = mlb.transform(y_test)
    print('Data processed')


"""#### Model"""

input = Input(shape=(no_of_leads, 1000, 1))

conv1 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1))(input)
batch1 = BatchNormalization()(conv1)
relu1 = ReLU()(batch1)

conv2 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1))(relu1)
batch2 = BatchNormalization()(conv2)
relu2 = ReLU()(batch2)
drop2 = Dropout(rate=0.1)(relu2)
conv2 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2))(drop2)

max1 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(relu1)
conv_ = Conv2D(64, (1, 1))(max1)
conc1 = Add()([conv2, conv_])

batch3 = BatchNormalization()(conc1)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1))(drop3)
batch3 = BatchNormalization()(conv3)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2))(drop3)

max2 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc1)
conc2 = Add()([conv3, max2])

batch3 = BatchNormalization()(conc2)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 1))(drop3)
batch3 = BatchNormalization()(conv3)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 2))(drop3)

max3 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc2)
conv_ = Conv2D(128, (1, 1))(max3)
conc3 = Add()([conv3, conv_])

batch3 = BatchNormalization()(conc3)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 1))(drop3)
batch3 = BatchNormalization()(conv3)
relu3 = ReLU()(batch3)
drop3 = Dropout(rate=0.1)(relu3)
conv3 = Conv2D(filters=128, kernel_size=(1, 5), strides=(1, 2))(drop3)

max4 = MaxPooling2D(pool_size=(1, 9), strides=(1, 2))(conc3)
conc4 = Add()([conv3, max4])

conv3 = Conv2D(filters=128, kernel_size=(no_of_leads, 1))(conc4)
X = BatchNormalization()(conv3)
X = ReLU()(X)
X = GlobalAveragePooling2D()(X)

X = Flatten()(X)
print(X.shape)

X = Dense(units=128, kernel_regularizer=tf.keras.regularizers.L2(0.005))(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Dropout(rate=0.1)(X)

X = Dense(units=64, kernel_regularizer=tf.keras.regularizers.L2(0.009))(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Dropout(rate=0.15)(X)
print('Added 2 fully connected layers')

output = Dense(no_of_classes, activation='sigmoid')(X)
model = Model(inputs=input, outputs=output)

print(model.summary())


"""#### Train Model"""

early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reducelr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
callback = [early, reducelr]

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss=losses.BinaryCrossentropy(),
              metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve='ROC', multi_label=True)])

history = model.fit(x_train, y_train, validation_split=0.12,
                    epochs=100, batch_size=32, callbacks=callback)


"""##### Save Model"""

save_path = path + 'models\\'
os.makedirs(save_path, exist_ok=True)
model.save(save_path + "First_Paper.h5")
print("Model saved at:", save_path + "First_Paper.h5")
