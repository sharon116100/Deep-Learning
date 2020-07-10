# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:00:22 2020

@author: user
"""

import numpy as np
import pandas as pd
import glob
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torch.utils.data as Data
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # input shape:(1, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
#            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape: (32, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=3, stride=1, padding=1),
#            nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape: (64, 32, 32)
        )
        self.out = nn.Linear(in_features=64 * 32 * 32, out_features=3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten
        x = x.view(x.size()[0], -1) # (batch size, 28*28)
        output = self.out(x)
        return output
    
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
training_image = []
training_label = []
testing_image = []
testing_label = []

for i in range(len(train)):
    try:
#        image = Image.open("images/images/"+train['filename'][i])
#        image = image.numpy()
        image = cv2.imread("images/images/"+train['filename'][i])
        image_face = cv2.resize(image[train['ymin'][i]:train['ymax'][i], 
                                      train['xmin'][i]:train['xmax'][i]], (128,128))
        training_image.append(image_face)
        if train['label'][i] == 'good':
            training_label.append(0)
        elif train['label'][i] == 'none':
            training_label.append(1)
        else:
            training_label.append(2)
#        training_label.append(train['label'][i])
    except:
        pass
for i in range(len(test)):
    try:
#        image = Image.open("images/images/"+test['filename'][i])
#        image = image.numpy()
        image = cv2.imread("images/images/"+test['filename'][i])
        image = image[test['ymin'][i]:test['ymax'][i],test['xmin'][i]:test['xmax'][i]]        
        image_face = cv2.resize(image, (128,128))
#        cv2.imshow(str(i), image_face)
#        cv2.waitKey(0)
        testing_image.append(image_face)
        if test['label'][i] == 'good':
            testing_label.append(0)
        elif test['label'][i] == 'none':
            testing_label.append(1)
        else:
            testing_label.append(2)
#        testing_label.append(test['label'][i])
    except:
        pass

training_image = np.array(training_image)
training_label = np.array(training_label)
testing_image = np.array(testing_image)
testing_label = np.array(testing_label)

training_X = torch.from_numpy(training_image.transpose((0,3,1,2))) #transpose = 改training_image資料的位置
training_X = training_X.float().div(255) # 改成float且範圍改到(0,1)
training_y = torch.from_numpy(training_label).long()
testing_X = torch.from_numpy(testing_image.transpose((0,3,1,2)))
testing_X = testing_X.float().div(255)
testing_y = torch.from_numpy(testing_label).long()
training_dataset = Data.TensorDataset(training_X, training_y)
testing_dataset = Data.TensorDataset(testing_X, testing_y)

epoches = 20
batch_size = 50
lr = 0.001
training_loader = Data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
testing_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

net1 = CNN()
print(net1)
optimizer = torch.optim.Adam(net1.parameters(), lr=lr)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
history_loss = []
history_train_acc = []
history_test_acc = []
for epoch in range(epoches):
    print('Epoch:', epoch)
    train_loss = 0
    train_correct = 0
    train_total = 0
    for step, (b_x, b_y) in enumerate(training_loader):   # 分配 batch data, normalize x when iterate train_loader
        output = net1(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()     
           # apply gradients
        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += b_y.size(0)
        train_correct += predicted.eq(b_y).sum().item()
    history_loss.append(train_loss/len(training_loader))
    history_train_acc.append(train_correct/train_total)
    print('【Training】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( train_loss/len(training_loader), 100.*(train_correct/train_total), train_correct, train_total ))

    test_loss = 0
    test_correct = 0
    test_total = 0
    for step, (b_x, b_y) in enumerate(testing_loader):   # 分配 batch data, normalize x when iterate train_loader
        output = net1(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step

        test_loss += loss.item()
        _, predicted = output.max(1)
        test_total += b_y.size(0)
        test_correct += predicted.eq(b_y).sum().item()
    history_test_acc.append((test_correct/test_total))
    print('【Testing】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( test_loss/len(testing_loader), 100.*(test_correct/test_total), test_correct, test_total ))

# (2)  Plot the learning curve and the accuracy rate of training and test data
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


# (3) accuracy of each classes
#good_train_sample = training_X[training_y == 0]

#bad_train_sample = training_X[training_y == 2]
#good_test_sample = testing_X[testing_y == 0]

#bad_test_sample = testing_X[testing_y == 2]
eachclass = [0,1,2]
for i in eachclass:
    if i == 0:
        print('good:')
    elif i == 1:
        print('none')
    else:
        print('bad')
    train_min_batches = [training_X[k:k+batch_size] for k in range(0,len(training_X[training_y == i]),batch_size)]
    train_class_output = []
    train_class_pred = []
    for j in train_min_batches:
        train_class_output = net1(j)
        train_class_pred.extend(torch.max(train_class_output, 1)[1].data.numpy().squeeze())
#    train_class_pred = train_class_output.max(1)
    correct = sum(int(a == i) for a in train_class_pred)
    print('【Training】Acc: %.3f%%' % (100.*(correct/len(train_class_pred))))
    
    test_min_batches = [testing_X[k:k+batch_size] for k in range(0,len(testing_X[testing_y == i]),batch_size)]
    test_class_output = []
    test_class_pred = []
    for j in test_min_batches:
        test_class_output = net1(j)
#        test_class_output.append(net1(j))
        test_class_pred.extend(torch.max(test_class_output, 1)[1].data.numpy().squeeze())
    correct = sum(int(a == i) for a in test_class_pred)
    print('【Testing】Acc: %.3f%%' % (100.*(correct/len(test_class_pred))))

none_train_sample = training_image[training_label == 1]
none_test_sample = testing_image[testing_label == 1]
fliplr_train_image = np.fliplr(none_train_sample)
flipud_train_image = np.flipud(none_train_sample)
fliplr_test_image = np.fliplr(none_test_sample)
flipud_test_image = np.flipud(none_test_sample)

new_training_image = np.concatenate((training_image, fliplr_train_image), axis = 0)
new_training_image = np.concatenate((new_training_image, flipud_train_image), axis = 0)
new_testing_image = np.concatenate((testing_image, fliplr_test_image), axis = 0)
new_testing_image = np.concatenate((new_testing_image, flipud_test_image), axis = 0)
new_training_label = np.concatenate((training_label, np.ones(2*len(none_train_sample))), axis = 0)
new_testing_label = np.concatenate((testing_label, np.ones(2*len(none_test_sample))), axis = 0)

training_X = torch.from_numpy(new_training_image.transpose((0,3,1,2))) #transpose = 改training_image資料的位置
training_X = training_X.float().div(255) #改成float且範圍改到(0,1)
training_y = torch.from_numpy(new_training_label).long()
testing_X = torch.from_numpy(new_testing_image.transpose((0,3,1,2)))
testing_X = testing_X.float().div(255)
testing_y = torch.from_numpy(new_testing_label).long()
training_dataset = Data.TensorDataset(training_X, training_y)
testing_dataset = Data.TensorDataset(testing_X, testing_y)
#cv2.imshow('a', flip_image[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()