# -*- coding: big5 -*-
"""
Created on Wed Jun 24 11:26:47 2020

@author: user
"""

import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import re
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device: %s"%device)

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

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
allowPOS = ['a','v','n','d']
"""
train_data_jieba = train_data.copy()
print('start train')
for i in range(len(train_data)):
    if i%1000==0:
        print(i)
    line = train_data['title'].iloc[i]
    word = re.sub("[A-Za-z0-9\[\–\-\`\～\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\\n\\xa0]", "",line)
    train_data_jieba['title'].iloc[i] = word
print('finish train')

test_data_jieba = test_data.copy()
print('start test')
for i in range(len(test_data)):
    if i%1000==0:
        print(i)
    line = test_data['title'].iloc[i]
    word = re.sub("[A-Za-z0-9\[\–\-\`\～\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\\n\\xa0]", "",line)
    test_data_jieba['title'].iloc[i] = word
print('finish test')

test_data_jieba.to_csv('test_data_jieba.csv')
train_data_jieba.to_csv('train_data_jieba.csv')


#Title Jieba
jieba.analyse.set_stop_words('stopwords.txt')
title_train = []
title_test = []
for i in range(len(train_data)):
    try: #jieba
        title_train.append(jieba.analyse.extract_tags(train_data_jieba['title'].loc[i], withWeight=False, allowPOS=allowPOS))
    except:
        title_train.append([])
    if i%1000==0:
        print(title_train[i])
for i in range(len(test_data)):
    try: #jieba
        title_test.append(jieba.analyse.extract_tags(test_data_jieba['tile'].loc[i], withWeight=False, allowPOS=allowPOS))
    except:
        title_test.append([])
    if i%1000==0:
        print(title_test[i])
train_data_jieba['title_jieba'] = title_train
test_data_jieba['title_jieba'] = title_test
train_data_jieba.to_csv('train_data_jieba.csv')
test_data_jieba.to_csv('test_data_jieba.csv')


train_data_jieba = pd.read_csv('train_data_jieba.csv')
test_data_jieba = pd.read_csv('test_data_jieba.csv')

jieba_train = []
jieba_test = []
content_train = train_data_jieba['title_jieba'].tolist()
content_test = test_data_jieba['title_jieba'].tolist()
for i in range(len(train_data)):
    if i%1000==0:
        print(i)    
    keyword = train_data_jieba['keyword'].iloc[i]
    if pd.isnull(keyword):
        keyword = []
    else:
        keyword = keyword.split(",")
    #keyword.extend(content_train[i][2:-2].split("', '") )
    keyword.extend(str(content_train[i])[2:-2].split("', '"))
    jieba_train.append(keyword)
#print(jieba_train)

for i in range(len(test_data)):
    if i%1000==0:
        print(i)
    keyword = test_data_jieba['keyword'].iloc[i]
    if pd.isnull(keyword):
        keyword = []
    else:
        keyword = keyword.split(",")
    keyword.extend(content_test[i][2:-2].split("', '") )
    jieba_test.append(keyword)
          
train_data_jieba['jieba'] = jieba_train
test_data_jieba['jieba'] = jieba_test
train_data_jieba.to_csv('train_data_jieba.csv')
test_data_jieba.to_csv('test_data_jieba.csv')
"""
print("start word embedding")
# jieba分詞轉word2vec向量
from gensim.models.word2vec import Word2Vec
corpus = pd.concat([train_data_jieba.jieba, test_data_jieba.jieba]).sample(frac = 1)
model = Word2Vec(corpus, size=400, window=9, min_count=5, sg=1, iter=10)
model.save("word2vec.model")



print("start dataloader")
training_data_x = np.array(train_data_jieba['jieba'])
training_data_y =  np.array(train_data_jieba['label']).astype('float64')
#training_data_y = training_data_y.astype('float64')
training_data_x = (training_data_x - np.min(training_data_x)) / (np.max(training_data_x) - np.min(training_data_x))
testing_data_x = np.array(test_data_jieba['jieba'])
testing_data_x = (testing_data_x - np.min(testing_data_x)) / (np.max(testing_data_x) - np.min(testing_data_x))

training_data_x = torch.Tensor(training_data_x)
training_data_y = torch.Tensor(training_data_y)
testing_data_x = torch.Tensor(testing_data_x)

training_dataset = Data.TensorDataset(training_data_x, training_data_y)
testing_dataset = Data.TensorDataset(testing_data_x, np.zeros((len(testing_data_x))))

validation_split = .2
dataset_size = len(training_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Hyper Parameters
EPOCH = 500
BATCH_SIZE = 64
TIME_STEP = 1
INPUT_SIZE = None
LR = 0.0001

train_loader = Data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
valid_loader = Data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
test_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)


net = RNN()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.BCELoss()   # the target label is not one-hotted

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
        b_x = b_x.to(device)
        b_y = b_y.to(device)
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
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = net(b_x)
            loss = loss_func(output, b_y.squeeze())   # cross entropy loss
            valid_loss += loss.item()
            predicted = torch.gt(output, 0.5)
    #        _, predicted = output.max(1)
    #        print(output, predicted)
    #        predicted = torch.max(output, 1)[1].data.numpy()
            valid_total += b_y.size(0)
            valid_correct += predicted.long().eq(b_y.long()).sum().item()
    if epoch%50 == 0:
        torch.save(net, './model/lstm_model_{}.pth'.format(epoch))
    history_loss.append(train_loss/len(train_loader))
    history_train_acc.append(train_correct/train_total)
    history_test_acc.append((valid_correct/valid_total))
    print(' EPOCH: %d, step: %d【Training】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(train_correct/train_total), train_correct, train_total ))
    print(' EPOCH: %d, step: %d【Validation】Loss: %.3f | Acc: %.3f%% (%d/%d) ' % (epoch, step , train_loss/len(train_loader), 100.*(valid_correct/valid_total), valid_correct, valid_total ))
    
