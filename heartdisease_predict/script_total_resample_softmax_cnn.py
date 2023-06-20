# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:51:57 2023

@author: HuangAlan
"""

import numpy as np
import pandas as pd
SEED = 42

# %% initial data
path_csv = 'heart_2020_cleaned.csv'
data = pd.read_csv(path_csv, header=0)

# ----- encoding age and health -----
health_bin =['Poor', 'Fair', 'Very good', 'Good', 'Excellent']
for org, new in zip(health_bin, range(len(health_bin))):
    data['GenHealth']=data['GenHealth'].replace(org, new)
    
# ----- race vs disease -----
get_disease = {}
race_count = 0
race_bin = ['American Indian/Alaskan Native' ,'Asian', 'Black', 'Hispanic', 
            'Other', 'White']
for i_race in race_bin:
    tmp = data[data['Race']==i_race]
    n, c = np.unique(tmp['HeartDisease'], return_counts=True)
    get_disease[i_race] = c[0]/(c[0]+c[1])
    if np.round(get_disease[i_race], 2) >= 0.9:
        race_count += 1
    
# ----- encoding -----
trans_rule = {}
for i_col in data.columns:
    try:
        _data_trans = data[i_col].astype(np.float64)
    except:
        _cate = np.unique(data[i_col])
        _rule = {}
        for i, i_ele in enumerate(_cate):
            _rule.update({i_ele: i})
        trans_rule[i_col] = _rule
        _data_trans = data[i_col].map(_rule).astype(int)
    if i_col == data.columns[0]:
        data_numl = _data_trans
    else:
        data_numl = pd.concat([data_numl, _data_trans], axis=1)

# %% split dataset 7:3 and drop feature
class_p = data_numl.loc[data_numl['HeartDisease']==1]
class_n = data_numl.loc[data_numl['HeartDisease']==0]
data_tra = pd.concat([class_p.iloc[:int(len(class_p)*0.7),:],
                      class_n.iloc[:int(len(class_n)*0.7),:]])
data_tes = pd.concat([class_p.iloc[int(len(class_p)*0.7):,:],
                      class_n.iloc[int(len(class_n)*0.7):,:]])

# ----- select feature by data_tra -----
drop_fea_bool = (np.tril(data_tra.corr().abs(), k=-1)>0.8).any(axis=1) # 0.8 corr thold
drop_fea_index = list(set(np.where(drop_fea_bool)[0]))
drop_fea = data_tra.columns[drop_fea_index]
ratio = pd.DataFrame([], 
                     columns=['AlcoholDrinking', 'Asthma', 'KidneyDisease',
                              'PhysicalActivity', 'Smoking', 'Stroke', 
                              'SkinCancer'],
                     index=[0,1])
for i_group in ratio.index:
    _data_tra_g = data_tra[data_tra['HeartDisease']==i_group]
    for i_col in ratio.columns:
        n, c = np.unique(_data_tra_g[i_col],return_counts=True)
        ratio.loc[i_group, i_col] = (c[1]/(c[0]+c[1]))
drop_fea = drop_fea.append(ratio.columns[(ratio>0.5).all()])

# ----- drop -----
for i_col in drop_fea:
    data_tra.pop(i_col)
    data_tes.pop(i_col)
if race_count == len(race_bin):
    data.pop('Race')

# %% data over sample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample

# ----- select normal data -----
data_tra_p = data_tra.loc[(data_tra['HeartDisease']==1)]
data_tra_n = data_tra.loc[(data_tra['HeartDisease']==0)]
data_tra_p = resample(data_tra_p, n_samples=len(data_tra_n), 
                      random_state=SEED)
data_tra_sel = pd.concat([data_tra_p, data_tra_n])

# ----- seperate train/valid -----
data_tra, data_val = train_test_split(data_tra_sel, test_size=0.2, 
                                      stratify=data_tra_sel['HeartDisease'], 
                                      random_state=SEED)
 
# ----- resort data_tra -----
data_tra_p = data_tra[data_tra['HeartDisease']==1]
data_tra_n = data_tra[data_tra['HeartDisease']==0]
n, m = np.shape(data_tra_p)
data_tra_sort = np.empty((2*n, m))
data_tra_sort[0::2] = data_tra_p.values
data_tra_sort[1::2] = data_tra_n.values
data_tra_sort = pd.DataFrame(data_tra_sort, columns=data_tra_sel.columns)

# ----- label one hot -----
label_tra = to_categorical(np.array(data_tra_sort.pop('HeartDisease')), 
                            num_classes=2, dtype='float32')
label_val = to_categorical(np.array(data_val.pop('HeartDisease')), 
                            num_classes=2, dtype='float32')
label_tes = np.array(data_tes.pop('HeartDisease'))

# ----- normalized -----
sc = MinMaxScaler()
data_tra = sc.fit_transform(data_tra_sort)
data_val = sc.transform(data_val)
data_tes = sc.transform(data_tes)
_, n_dim = np.shape(data_tra)

 # %% classify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.compat import v1
import random
import os

# ----- ensure the result can be recurrent -----
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
v1.set_random_seed(SEED) 

# ----- 1D CNN ------
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(n_dim,1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),
    
    Conv1D(64, kernel_size=3, activation='relu'),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5, seed=42),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
    ])

#---- compile ----- 
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy', 
              metrics=[metrics.Recall(), metrics.BinaryAccuracy()])

# ----- train -----
condi = {'monitor_1': 'val_recall', 'mode_1': 'max',
         'monitor_2': 'val_loss', 'mode_2': 'min',
         'monitor_3': 'val_binary_accuracy'}
early_stop = EarlyStopping(monitor=condi['monitor_1'], mode=condi['mode_1'], 
                           patience=20, min_delta=1e-2, 
                           restore_best_weights=True)
check_point = ModelCheckpoint('train_record/check_point_cnn.h5', 
                              monitor=condi['monitor_1'], mode=condi['mode_1'],
                              save_best_only=True, verbose=1)

net_hist = model.fit(x=data_tra, y=label_tra, epochs=100, batch_size=10000,
                     validation_data=(data_val, label_val),
                     callbacks=[early_stop, check_point], shuffle=False, 
                     verbose=1)

# %% test
from tensorflow.keras.models import load_model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

# ----- performance -----
best_model = load_model('train_record/check_point_cnn.h5')
tra_result = best_model.evaluate(data_tra, label_tra)
print(f'tra_loss: {tra_result[0] :.4f}, tra_recall: {tra_result[1] :.4f},'+
      f' tra_acc: {tra_result[2] :.4f}')

tes_prob = best_model.predict(data_tes, verbose=0)[:,1]
tes_fpr, tes_tpr, prob_thd = roc_curve(label_tes, tes_prob)
tes_auc = auc(tes_fpr, tes_tpr)
ind = np.where(abs(prob_thd-0.5) == min(abs(prob_thd-0.5)))[0][0]
print(f'test_auc: {tes_auc :.4f}, test_tpr: {tes_tpr[ind] :.4f},'+
      f' test_tnr: {1-tes_fpr[ind] :.4f}')
