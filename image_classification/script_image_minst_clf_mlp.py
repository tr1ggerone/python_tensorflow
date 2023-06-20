# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:37:03 2023

@author: HuangAlan
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint

# ----- ensure the result can be recurrent -----
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED']=str(0)
tf.compat.v1.set_random_seed(0) 

# %% load data
(data_tra, label_tra), (data_tes, label_tes) =  mnist.load_data()
data_tra = np.expand_dims(data_tra.astype(np.float32)/255.0, axis=-1)
data_tes = np.expand_dims(data_tes.astype(np.float32)/255.0, axis=-1)

# ----- one-hot encoding -----
num_c = len(np.unique(label_tra))
label_tra = to_categorical(label_tra, num_classes=num_c)
label_tes = to_categorical(label_tes, num_classes=num_c)

# %% gen MLP and compile
# ----- method 1 (save as .h5 files) -----
input_shape = np.shape(data_tra)[1:]
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Flatten())
model.add(Dense(units=100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2, seed=42))
model.add(Dense(units=10, activation='softmax'))

# ----- method 2 (cannot save as .h5 files, cause no declare a Sequential()) -----
# class MLP(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
#     def call(self, inputs):
#         x = self.flatten(inputs)
#         x = self.dense1(x)
#         output = self.dense2(x)
#         return output
# model = MLP()

# ----- compile -----
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# %% fit
# ----- callback, early stop -----
early_stop = EarlyStopping(monitor='loss', mode='auto', 
                           patience=10, min_delta=1e-2, 
                           restore_best_weights=True)

# ----- callback, save training log -----
csv_logger = CSVLogger('train_record_mlp/model_mnist_mlp_training.csv',
                             separator=',', append=False)

# ----- callback, save the good result -----
check_point = ModelCheckpoint(filepath='train_record_mlp/model_mnist_mlp.h5',
                              save_weights_only=False, monitor='loss',
                              mode='auto', save_best_only=True)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data_tra, label_tra, epochs=100, batch_size=256, 
          callbacks=[early_stop, csv_logger], verbose=1)

# %% load model and test
model_pretrain = load_model('train_record_mlp/model_mnist_mlp.h5')
model_pretrain.summary()
model_pretrain.get_layer('dropout').training = False
test_loss, test_acc = model_pretrain.evaluate(data_tes, label_tes)
