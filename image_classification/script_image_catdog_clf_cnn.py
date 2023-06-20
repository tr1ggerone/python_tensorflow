# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:46:32 2023

@author: HuangAlan

img generator ref: 
    https://keras.io/zh/preprocessing/image/ 
    https://blog.csdn.net/dugudaibo/article/details/87719078
"""
import os
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model

# ----- ensure the result can be recurrent -----
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED']=str(0)
tf.compat.v1.set_random_seed(0) 

# %% load img
def load_img(path_dir, img_size):
    from tensorflow.keras.utils import load_img
    from tqdm import tqdm
    
    # ----- load -----
    img_set = []
    label_set = []
    for i_dir in os.listdir(path_dir):
        if i_dir.lower() == 'dog' or 'dog' in i_dir.lower():
            _path_set = os.path.join(path_dir, i_dir)
            _label = 0
        elif i_dir.lower() == 'cat' or 'cat' in i_dir.lower():
            _path_set = os.path.join(path_dir, i_dir)
            _label = 1
        else:
            break
        for i_pic in tqdm(os.listdir(_path_set)):
            _path_img = os.path.join(_path_set, i_pic)
            try:
                img = np.array(load_img(_path_img,
                                        target_size=img_size,
                                        color_mode='rgb')).astype(np.float32)
                img = img/255.0 # normalize
                img_set.append(img)
                label_set.append(_label)
            except:
                print('fail in load', i_pic)
                os.remove(_path_img)
    return np.array(img_set), np.array(label_set)

# %% define cnn model
def define_model(img_size):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # ---- block 1 -----
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same', input_shape=(img_size[0], img_size[1], 3)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        # ---- block 2 -----
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
    
        # ---- block 3 -----
        Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        # ---- block 4 -----
        Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        BatchNormalization(),
        
        # ---- full connect layer ----
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2, seed=42),
        Dense(64, activation='relu'),
        Dropout(0.1, seed=42),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    
    #---- compile ----- 
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def plot_train_history(history):
    import matplotlib.pyplot as plt
    
    # ----- plot loss -----
    plt.figure()
    plt.subplot(211)
    plt.title('binary_crossentropy')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='valid')
    
    # ----- plot accuracy -----
    plt.subplot(212)
    plt.title('binary_accuracy')
    plt.plot(history.history['binary_accuracy'], color='blue', label='train')
    plt.plot(history.history['val_binary_accuracy'], color='orange', label='valid')

def plot_predict_result(input_img, input_prob, img_label):
    import matplotlib.pyplot as plt
    # ----- set parameter -----
    plt.figure()
    i_pic = 0
    pic_count = 25 if len(input_img) >= 25 else len(input_img)
    plt.gcf().set_size_inches(12, 9)
    
    # ----- plot -----
    for i_img in range(pic_count):
        ax = plt.subplot(5, 5, i_pic+1)
        ax.imshow(input_img[i_img])
        _judge = 1 if input_prob[i_img] > 0.5 else 0
        if _judge == img_label[i_img]:
            title = f'True, label:{img_label[i_img]}, pred:{_judge}'
        else:
            title = f'Fail, label:{img_label[i_img]}, pred:{_judge}'
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        i_pic += 1
    plt.tight_layout()
    
# %% train cnn and save
img_size = (100,100)
if input('re-train:') != '':
    # ----- gen dataset -----
    path_dir = 'C:/Users/HNC-3090/Desktop/PetImages'
    img_set, label_set = load_img(path_dir, img_size)
    
    # ----- seperate dataset -----
    data_tra, data_tes, label_tra, label_tes = train_test_split(
        np.array(img_set), np.array(label_set), test_size=0.2, random_state=42, 
        stratify=label_set)
    data_tra, data_val, label_tra, label_val = train_test_split(
        data_tra, label_tra, test_size=0.2, random_state=42, 
        stratify=label_tra)
    
    # ----- set callback -----
    # early_stop = EarlyStopping(monitor='val_loss', mode='auto', patience=5, 
    #                            min_delta=0.01, restore_best_weights=True)
    check_point = ModelCheckpoint('train_record_cnn/cnn_check_point.h5', 
                                  monitor='val_loss', save_best_only=True, 
                                  mode='min', verbose=1)
    
    # ----- train -----
    model = define_model(img_size)
    net_hist = model.fit(x=data_tra, y=label_tra, epochs=10, batch_size=48, 
                          validation_data=(data_val,label_val), 
                          callbacks=[check_point], shuffle=False, verbose=1)
    plot_train_history(net_hist)

# %% test with saved model
best_model = load_model('train_record_cnn/cnn_check_point.h5')
for i_name in ['batch_normalization', 'dropout', 'dropout_1']:
    try:
        best_model.get_layer(i_name).training = False
    except:
        pass
try:
    _, val_acc = best_model.evaluate(x=data_val, y=label_val, verbose=1)
    _, tes_acc = best_model.evaluate(x=data_tes, y=label_tes, verbose=1)
except NameError:
    print('dismiss re-train')

# ----- independent test -----
tes_img, tes_label = load_img('test_set', img_size)
_, tes_acc = best_model.evaluate(x=tes_img, y=tes_label, verbose=1)
tes_prob = best_model.predict(tes_img, verbose=0).ravel()
plot_predict_result(tes_img[:70], tes_prob[:70], tes_label[:70])
plot_predict_result(tes_img[70:], tes_prob[70:], tes_label[70:])
