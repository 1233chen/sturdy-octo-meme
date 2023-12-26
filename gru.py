import os
import math
import numpy
from openpyxl import load_workbook
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pickle
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import random
import keras
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from keras.models import Model
from keras.models import load_model
from keras import backend, layers, models, utils
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout,Flatten,GlobalAveragePooling1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Reshape
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
from torch.utils.data import random_split
from transformers import BertTokenizer, BertForSequenceClassification

def standarded(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    return (data-mean)/std

feature=[[],[],[],[]]
label=[]

for l1 in range(4,7):
    for l2 in range(2,3):
        file_path_feature=f"./{l1}_{l2}_final.txt"
        print(file_path_feature)
        with open(file_path_feature,'r') as file:
            line=file.read()
            line=line.split()
            sz=len(line)
            for i in range(sz):
                if (i%5!=4):feature[i%5].append(int(line[i]))
                else:
                    x=int(line[i])
                    if (x==1):
                        for j in range(4):feature[j].pop()
                    else:
                        if (x==2):x=1
                        label.append(x)


feature=np.array(feature)
label=np.array(label)
feature=np.transpose(feature,(1,0))

#归一化一下 只考虑比例
#data_normalized = (feature - np.min(feature, axis=0)) / (np.max(feature, axis=0) - np.min(feature, axis=0))
#feature=data_normalized

feature=standarded(feature)

all_len=feature.shape[0]

print(feature.shape)

#数据预处理
final_feature=[]
final_label=[]

window_size=128
for i in range(all_len):
    us=True
    if (i+i+window_size-1>=all_len):break
    for j in range(i+1,i+window_size):
        if (label[j]!=label[i]):
            us=False
            break
    if (us):
        for j in range(i,i+window_size):
            for k in feature[j]:
                final_feature.append(k)
        final_label.append(label[i])
final_feature=np.array(final_feature)
final_feature=final_feature.reshape(-1,window_size,4)
final_label=np.array(final_label)

totlen=22000
tmplen=len(final_label)
one=[]
zero=[]
for i in range(tmplen):
    if (final_label[i]==0):zero.append(i)
    else:one.append(i)
random.shuffle(one)
random.shuffle(zero)

addone=totlen-len(one)
addzero=totlen-len(zero)

for i in range(addzero):
    tmp=final_feature[zero[i]]
    tmp=tmp.reshape(-1,window_size,4)
    final_feature=np.concatenate([final_feature,tmp])
    tmp=[0]
    tmp=np.array(tmp)
    final_label=np.concatenate([final_label,tmp])

for i in range(addone):
    tmp=final_feature[one[i]]
    tmp=tmp.reshape(-1,window_size,4)
    final_feature=np.concatenate([final_feature,tmp])
    tmp=[1]
    tmp=np.array(tmp)
    final_label=np.concatenate([final_label,tmp])


print(final_feature.shape)
print(final_label.shape)

#dataset
class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]

dataset=MyDataset(final_feature,final_label)

total_size=len(dataset)
train_size=int(0.8*total_size)
val_size=total_size-train_size
train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
train_data=DataLoader(dataset,shuffle=True,batch_size=32)
test_data=DataLoader(val_dataset,shuffle=True,batch_size=32)
        
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
idx=np.random.permutation(len(final_feature))
tmp1=[final_feature[i] for i in idx]
tmp2=[final_label[i] for i in idx]
final_feature=tmp1
final_label=tmp2
print("shuffle done")
train_len=int(0.8*len(final_feature))
train_feature=final_feature[0:train_len]
train_label=final_label[0:train_len]
train_feature=np.array(train_feature)
train_label=np.array(train_label)

evaluate_feature=final_feature[train_len:]
evaluate_label=final_label[train_len:]
evaluate_feature=np.array(evaluate_feature)
evaluate_label=np.array(evaluate_label)
model=Sequential()
model.add(GRU(128,input_shape=(window_size,4),activation='tanh',return_sequences=False))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_feature,
                    train_label,
                    batch_size=32,
                    epochs=30,
                    validation_split=0.2,
                    verbose=2)

loss, accuracy = model.evaluate(evaluate_feature, evaluate_label)
print(loss,accuracy)