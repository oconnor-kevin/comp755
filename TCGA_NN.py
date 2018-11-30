#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:48:37 2018

@author: kevinoconnor
"""
import pandas as pd
import numpy as np
from random import shuffle
from keras import models as models
from keras import layers
from keras.layers import *
from keras.layers.core import *
from keras import optimizers

weight_decay = 0.0005
num_classes = 5

# Read in data.
dat_luma = np.transpose(pd.read_csv('/Users/kevinoconnor/Documents/School/COMP_755/luma.csv', header=None).values)
dat_lumb = np.transpose(pd.read_csv('/Users/kevinoconnor/Documents/School/COMP_755/lumb.csv', header=None).values)
dat_basal = np.transpose(pd.read_csv('/Users/kevinoconnor/Documents/School/COMP_755/basal.csv', header=None).values)
dat_her2 = np.transpose(pd.read_csv('/Users/kevinoconnor/Documents/School/COMP_755/her2.csv', header=None).values)
dat_normal = np.transpose(pd.read_csv('/Users/kevinoconnor/Documents/School/COMP_755/normal.csv', header=None).values)

x = np.concatenate((dat_luma, dat_lumb, dat_basal, dat_her2, dat_normal))
y = np.array([0]*dat_luma.shape[0] + [1]*dat_lumb.shape[0] + [2]*dat_basal.shape[0] + [3]*dat_her2.shape[0] + [4]*dat_normal.shape[0])

# Randomly split data into train and test.
inds = [i for i in range(len(labels))]
shuffle(inds)
train_inds = inds[:round(0.8*len(labels))]
test_inds = inds[round(0.8*len(labels)):]
x_train = x[train_inds, :]
x_test = x[test_inds, :]
y_train = y[train_inds]
y_test = y[test_inds]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Build model.
model = models.Sequential()
#model.add(Dense(1024,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1024, input_shape=(x.shape[1],)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#model.add(Dense(2048,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#model.add(Dense(1024,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    batch_size=20,
                    validation_data=(x_test, y_test))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

