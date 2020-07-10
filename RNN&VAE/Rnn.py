# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:25:11 2020

@author: user
"""

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pygal.maps.world
from pygal.maps.world import COUNTRIES
import seaborn as sn
import pandas as pd
import numpy as np

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     
            input_size=19,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
        )

        self.L1 = nn.Linear(256, 128)
        self.L2 = nn.Linear(128, 32)
        self.L3 = nn.Linear(32, 8)
        self.out = nn.Linear(8, 1)
#        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 用全0的 state

        out = self.L1(r_out[:, -1, :])
        out = self.L2(out)
        out = self.L3(out)
        out = self.sigmoid(self.out(out))
#        out = self.sigmoid(self.out(r_out[:, -1, :]))
#        out = self.out(r_out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(     
            input_size=19,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
        )

        self.L1 = nn.Linear(256, 128)
        self.L2 = nn.Linear(128, 32)
        self.L3 = nn.Linear(32, 8)
        self.out = nn.Linear(8, 1)
#        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r_out, (h_n) = self.gru(x, None)

        out = self.L1(r_out[:, -1, :])
        out = self.L2(out)
        out = self.L3(out)
        out = self.sigmoid(self.out(out)) 
#        out = self.sigmoid(self.out(r_out[:, -1, :]))
#        out = self.out(r_out[:, -1, :])
        return out

def train_windows(df, ref_day=20, predict_day=1):
    X_train, Y_train = [], []
    for i in range(df.shape[0]-predict_day-ref_day):
        X_train.extend(np.array(df.iloc[i+1:i+ref_day]).T-np.array(df.iloc[i:i+ref_day-1]).T)
        Y_train.extend(np.array(df.iloc[i+ref_day:i+ref_day+predict_day]).T > np.array(df.iloc[i+ref_day-1:i+ref_day+predict_day-1]).T)
    return np.array(X_train), np.array(Y_train)

def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    return None


original_data = pd.read_csv('covid_19.csv')
original_data = original_data.rename(columns={'Unnamed: 0':'Country/Region'})


#first
correlation_data = original_data.drop(['Country/Region','Lat','Long'], axis =1)
df = correlation_data[2:].T
df.columns = original_data['Country/Region'][2:]
df = df.reset_index()
df = df.drop(['index'], axis = 1)
df = df.astype('int32')
corrMatrix = df.corr()
plt.figure(figsize=(100,100))
sn.heatmap(corrMatrix, annot=False)
plt.show()


#second

upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
high = [column for column in upper.columns if any(upper[column] > 0.95)]
high_corr = df[high]
#s = corrMatrix.unstack()
#so = s.sort_values(kind="quicksort", ascending=False)
#high_corr = so[so > 0.99]
#high_corr = high_corr[high_corr != 1]

training_data_x, training_data_y = train_windows(high_corr)
training_data_y = training_data_y.astype('float64')
training_data_x = (training_data_x - np.min(training_data_x)) / (np.max(training_data_x) - np.min(training_data_x))

training_data_x = torch.Tensor(training_data_x)
training_data_y = torch.Tensor(training_data_y)

training_dataset = Data.TensorDataset(training_data_x, training_data_y)
validation_split = .2
dataset_size = len(training_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Hyper Parameters
EPOCH = 50
BATCH_SIZE = 16
TIME_STEP = 1
INPUT_SIZE = 19
LR = 0.0001

train_loader = Data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = Data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

net = RNN()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.BCELoss()   # the target label is not one-hotted
history_loss = []
history_train_acc = []
history_test_acc = []

# training and testing
for epoch in range(EPOCH):
    train_loss = 0
    train_correct = 0
    train_total = 0
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    net.train()
    for step, (x, b_y) in enumerate(train_loader):   # gives batch data
        b_x = x.view(-1, TIME_STEP, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)

        output = net(b_x)               # rnn output
        loss = loss_func(output, b_y.squeeze())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)
        optimizer.step()                # apply gradients
        train_loss += loss.item()
        predicted = torch.gt(output, 0.5)
#        _, predicted = output.max(1)
#        print(output, predicted)
#        predicted = torch.max(output, 1)[1].data.numpy()
        train_total += b_y.size(0)
        train_correct += predicted.long().eq(b_y.long()).sum().item()
#        train_correct += float((predicted == b_y.numpy().astype(int).item()).sum())
#        print("predicted", predicted)
#        print("predicted.eq(b_y.long()).sum().item():",predicted.long().eq(b_y.long().squeeze()).sum().item())
    net.eval()
    with torch.no_grad():
        for step, (x, b_y) in enumerate(valid_loader):
            b_x = x.view(-1, TIME_STEP, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)
    
            output = net(b_x)
            loss = loss_func(output, b_y.squeeze())   # cross entropy loss
            valid_loss += loss.item()
            predicted = torch.gt(output, 0.5)
    #        _, predicted = output.max(1)
    #        print(output, predicted)
    #        predicted = torch.max(output, 1)[1].data.numpy()
            valid_total += b_y.size(0)
            valid_correct += predicted.long().eq(b_y.long()).sum().item()
    history_loss.append(train_loss/len(train_loader))
    history_train_acc.append(train_correct/train_total)
    history_test_acc.append((valid_correct/valid_total))
    print(' EPOCH: %d, step: %d【Training】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(train_correct/train_total), train_correct, train_total ))
    print(' EPOCH: %d, step: %d【Validation】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(valid_correct/valid_total), valid_correct, valid_total ))


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].set_title('Learning Cruve', color='r')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Cross Entropy')
ax[0].plot(history_loss)

ax[1].set_title('Accuracy', color='r')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_train_acc, label = 'train')
ax[1].plot(history_test_acc, label = 'test')
plt.legend(loc=1)

##third
net1 = GRU()
optimizer = torch.optim.Adam(net1.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.BCELoss()   # the target label is not one-hotted
history_loss_gru = []
history_train_acc_gru = []
history_test_acc_gru = []

# training and testing
for epoch in range(EPOCH):
    train_loss = 0
    train_correct = 0
    train_total = 0
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    net.train()
    for step, (x, b_y) in enumerate(train_loader):   # gives batch data
        b_x = x.view(-1, TIME_STEP, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)

        output = net1(b_x)               # rnn output
        loss = loss_func(output, b_y.squeeze())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)
        optimizer.step()                # apply gradients
        train_loss += loss.item()
        predicted = torch.gt(output, 0.5)
#        _, predicted = output.max(1)
#        print(output, predicted)
#        predicted = torch.max(output, 1)[1].data.numpy()
        train_total += b_y.size(0)
        train_correct += predicted.long().eq(b_y.long()).sum().item()
#        train_correct += float((predicted == b_y.numpy().astype(int).item()).sum())
#        print("predicted", predicted)
#        print("predicted.eq(b_y.long()).sum().item():",predicted.long().eq(b_y.long().squeeze()).sum().item())
    net.eval()
    with torch.no_grad():
        for step, (x, b_y) in enumerate(valid_loader):
            b_x = x.view(-1, TIME_STEP, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)
    
            output = net1(b_x)
            loss = loss_func(output, b_y.squeeze())   # cross entropy loss
            valid_loss += loss.item()
            predicted = torch.gt(output, 0.5)
    #        _, predicted = output.max(1)
    #        print(output, predicted)
    #        predicted = torch.max(output, 1)[1].data.numpy()
            valid_total += b_y.size(0)
            valid_correct += predicted.long().eq(b_y.long()).sum().item()
           
    history_loss_gru.append(train_loss/len(train_loader))
    history_train_acc_gru.append(train_correct/train_total)
    history_test_acc_gru.append((valid_correct/valid_total))
    print(' EPOCH: %d, step: %d【Training】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(train_correct/train_total), train_correct, train_total ))
    print(' EPOCH: %d, step: %d【Validation】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(valid_correct/valid_total), valid_correct, valid_total ))


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax[0].set_title('Learning Cruve', color='r')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Cross Entropy')
ax[0].plot(history_loss_gru)

ax[1].set_title('Accuracy', color='r')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(history_train_acc_gru, label = 'train')
ax[1].plot(history_test_acc_gru, label = 'test')
plt.legend(loc=1)

#fifth
eachcountry_x, eachcountry_y = train_windows(high_corr[-22:][:])
eachcountry_x = (eachcountry_x - np.min(eachcountry_x)) / (np.max(eachcountry_x) - np.min(eachcountry_x))
eachcountry_y = eachcountry_y.astype('float64')

eachcountry_x = torch.Tensor(eachcountry_x)
eachcountry_y = torch.Tensor(eachcountry_y)

eachcountry_dataset = Data.TensorDataset(eachcountry_x, eachcountry_y)
eachcountry_loader = Data.DataLoader(dataset=eachcountry_dataset, batch_size=178)

probability = []
for step, (x, b_y) in enumerate(eachcountry_loader):
    b_x = x.view(-1, TIME_STEP, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)

    output = net1(b_x)
    probability.extend(output.squeeze().tolist())

ascending = {}
decending = {}
for i in range(len(probability)):
    code = get_country_code(high[i])
    if probability[i] > 0.5:
        ascending[code] = probability[i]
    else:
        decending[code] = probability[i]
 
worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'the probability of each country'
worldmap_chart.add('ascending', ascending)
worldmap_chart.add('decending', decending)
worldmap_chart.render_to_file('world_population.svg') 