# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:50:50 2020

@author: user
"""

import numpy as np
import random
import pickle, gzip, urllib.request, json
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import itertools
#from sklearn.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

class DNN(object):
    def __init__(self,sizes): 
        self.num_layers = len(sizes) 
        self.sizes = sizes 
        self.weight = [np.random.randn(x, y) for (x, y) in zip(sizes[1:], sizes[:-1])]
        self.bias = [np.random.randn(x,1) for x in sizes[1:]]
#        self.weight = [np.zeros((x, y)) for (x, y) in zip(sizes[1:], sizes[:-1])]
#        self.bias = [np.zeros((x,1)) for x in sizes[1:]]
        self.training_error_rate = []
        self.testing_error_rate = []
        self.training_loss = []
        self.batches_latent = []
        self.latent = []
        self.latent_count = 0

    def SGD(self,training_data,min_batch_size,epoches,eta,testing_data=False):
        
        n = len(training_data)
        
#        self.training_error_rate = []
#        self.testing_error_rate = []
#        self.training_loss = []
#        evaluation_cost, evaluation_accuracy = [], []
#        training_cost, training_accuracy = [], []
        
        random.shuffle(training_data) # 打亂
        min_batches = [training_data[k:k+min_batch_size] for k in range(0,n,min_batch_size)] 
        
        for k in range(epoches): 
            total_loss = 0
            for min_batch in min_batches:
                total_loss += self.update_parameter(min_batch,eta)
            self.training_loss.append(total_loss / n)
            num_training = self.error_rate(training_data)
            num_testing = self.error_rate(testing_data)
            self.training_error_rate.append(num_training/n)
            self.testing_error_rate.append(num_testing/len(list(testing_data)))
            print("Loss:", total_loss / n)
            print("{0}th epoches: {1}/{2}({3})".format(k,num_training,n,num_training/n))
            print("{0}th epoches: {1}/{2}({3})".format(k,num_testing,len(list(testing_data)),num_testing/len(list(testing_data))))

    def forward(self,x): 
        #get acticativees values
        for layer, (w, b) in enumerate(zip(self.weight, self.bias)):
#            if layer == self.num_layers-1:
#                x = sigmoid(np.dot(w, x)+b)
#            else:
#                x = relu(np.dot(w, x)+b) 
            x = sigmoid(np.dot(w, x)+b)
        return x 

    def update_parameter(self,batch,lr): 
        ndeltab = [np.zeros(b.shape) for b in self.bias] 
        ndeltaw = [np.zeros(w.shape) for w in self.weight]
        batch_loss = 0
        for x,y in batch:
            deltab, deltaw, loss = self.backprop(x,y) 
            ndeltab = [nb +db for nb,db in zip(ndeltab,deltab)]
            ndeltaw = [nw + dw for nw,dw in zip(ndeltaw,deltaw)]
            batch_loss += loss
        self.bias = [b - lr * ndb/len(batch) for ndb,b in zip(ndeltab,self.bias)] 
        self.weight = [w - lr * ndw/len(batch) for ndw,w in zip(ndeltaw,self.weight)] 
        return batch_loss

    def backprop(self,x,y): 
        activation = x 
        activations = [x] 
        zs = [] 
        # feedforward 
        for layer, (w, b) in enumerate(zip(self.weight, self.bias)):            
            # print w.shape,activation.shape,b.shape 
            z = np.dot(w, activation) +b 
            zs.append(z)  #用來计算f(z)倒數
#            print("Layer:", layer)
#            print(len(self.weight))
            activation = sigmoid(z)
#            if layer == self.num_layers-1:
#                activation = sigmoid(z)
#            else:
#                activation = relu(z)
            # print 'activation',activation.shape 
            activations.append(activation) # 每層的输出结果
        self.latent_count+=1
        self.batches_latent.append(zs[-2])
        if self.latent_count == 10000:
#            print("self.latent:",len(self.latent))
#            print("self.batches_latent:", len(self.batches_latent))
            self.latent.append(self.batches_latent)
            self.latent_count = 0
            self.batches_latent = []
        delta = (activations[-1]-y)*dsigmoid(zs[-1])  #最後一層的delta,np.array乘,相同维度乘 
        deltaw = [np.zeros(w1.shape) for w1 in self.weight] #每一次將獲得的值作為列表形式给deltaw 
        deltab = [np.zeros(b1.shape) for b1 in self.bias] 
        # print 'deltab[0]',deltab[-1].shape 
        deltab[-1] = delta 
        deltaw[-1] = np.dot(delta,activations[-2].T)
        # print(activations[-1])
        
        for k in range(2,self.num_layers): 
#            delta = np.dot(self.weight[-k+1].T,delta) * drelu(zs[-k])
            delta = np.dot(self.weight[-k+1].T,delta) * dsigmoid(zs[-k])
            deltab[-k] = delta 
            deltaw[-k] = np.dot(delta,activations[-k-1].T) 
        return deltab,deltaw,Loss(activations[-1], y)

    def error_rate(self,data): 
        z = [(np.argmax(self.forward(x)),np.argmax(y)) for x,y in data] 
        zs = sum(int(a != b) for a,b in z) 
        return zs 

def sigmoid(x): 
    return 1.0/(1.0+np.exp(-x))
  
def dsigmoid(x): 
    z = sigmoid(x) 
    return z*(1-z)

def relu(x):
    return np.maximum(0,x)
#    x = (np.abs(x) + x) / 2.0
#    return x

def drelu(x):    
    x[x<=0] = 0
    x[x>0] = 1
    return x

def Loss(y, y_hat):
#    return np.nan_to_num(np.sum(-y_hat*np.log(y)-(1-y_hat)*np.log(1-y)))
    return -np.sum(y_hat * np.log(y, where=y>0))

def data_transform(t_d, te_d):    
    print('t_d',t_d['image'].shape)
    n = (np.array([np.reshape(x, (784, 1)) for x in t_d['image']])-128)/128 # 把(28,28)轉成（784,1）在normalization
    m = np.array([vectors(int(y)) for y in t_d['label']]) # （120000,1）改成（10,120000）
    
    train_data = list(zip(n[:10000],m[:10000]))
    validation_data = list(zip(n[10000:],list(t_d['label'][10000:])))
    
    n = (np.array([np.reshape(x, (784, 1)) for x in te_d['image']])-128)/128
    m = np.array([vectors(int(y)) for y in te_d['label']])
    test_data = list(zip(n, m))
    print('n',n.shape)
    
    return (train_data,validation_data, test_data) 

def vectors(y): 
    label = np.zeros((10,1)) 
    label[y] = 1.0
    return label

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#def tvectors(y): 
#    label = np.zeros((10)) 
#    label[y] = 1.0
#    return label 


#讀檔
train = np.load('train.npz')
test = np.load('test.npz')
train_data,validation_data, test_data = data_transform(train, test)

net1 = DNN([784, 30, 15, 2, 10])

min_batch_size = 50
lr = 0.05
epoches = 500
net1.SGD(train_data,min_batch_size,epoches,lr,test_data)
print("complete")

#1
new_x_axis = np.arange(0,500)

fig, ax = plt.subplots(1, 1)
ax.plot(new_x_axis, net1.training_loss)
ax.set_title('training loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Average cross entropy')

fig, ax = plt.subplots(1, 1)
ax.plot(new_x_axis, net1.training_error_rate)
ax.set_title('training error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error rate')

fig, ax = plt.subplots(1, 1)
ax.plot(new_x_axis, net1.testing_error_rate)
ax.set_title('testing error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error rate')
#urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
#with gzip.open('mnist.pkl.gz', 'rb') as f:
#    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#train_data,validation_data, test_data = data_transform(train_set, valid_set,test_set) 

#n = np.array([np.reshape(x, (784)) for x in train['image']])
#test_n = np.array([np.reshape(x, (784)) for x in test['image']])
#m = np.array([tvectors(int(y)) for y in train['label']])
#test_m = np.array([tvectors(int(y)) for y in test['label']])
#model = Sequential()
#model.add(Dense(units=30, input_dim=784, kernel_initializer='uniform'))
#model.add(Activation('relu'))
#model.add(Dense(units=15, kernel_initializer='uniform'))
#model.add(Activation('relu'))
#model.add(Dense(units=10, kernel_initializer='uniform'))
#model.add(Activation('sigmoid'))
#
#model.compile(loss = 'categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
#model.fit(x = n, y = m, epochs = 100, batch_size = 100, verbose = 1)
#scores = model.evaluate(x = test_n, y = test_m)

#3
colors = ['red', 'green', 'lightgreen', 'gray', 'cyan','blue','yellow','purple','orange','pink']

latent_feature_100 = np.array(net1.latent[100])
latent_feature_450 = np.array(net1.latent[300])
latent_feature_100_x = []
latent_feature_100_y = []
latent_feature_450_x = []
latent_feature_450_y = []
for i in range(10000):
    latent_feature_100_x.append(latent_feature_100[i][0])
    latent_feature_100_y.append(latent_feature_100[i][1])
    latent_feature_450_x.append(latent_feature_450[i][0])
    latent_feature_450_y.append(latent_feature_450[i][1])
label = np.array([(int(y)) for y in train['label']])
label = label[:10000]
cc = []
for i in label:
    cc.append(colors[i])

#legend 顏色處理
pop_0 = mpatches.Patch(color='red', label='0')
pop_1 = mpatches.Patch(color='green', label='1')
pop_2 = mpatches.Patch(color='lightgreen', label='2')
pop_3 = mpatches.Patch(color='gray', label='3')
pop_4 = mpatches.Patch(color='cyan', label='4')
pop_5 = mpatches.Patch(color='blue', label='5')
pop_6 = mpatches.Patch(color='yellow', label='6')
pop_7 = mpatches.Patch(color='purple', label='7')
pop_8 = mpatches.Patch(color='orange', label='8')
pop_9 = mpatches.Patch(color='pink', label='9')

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
scatter = ax.scatter(latent_feature_100_x, latent_feature_100_y, c = cc)
legend1 = ax.legend(handles=[pop_0,pop_1,pop_2,pop_3,pop_4,pop_5,pop_6,pop_7,pop_8,pop_9],loc="lower left", title="Classes")
ax.add_artist(legend1)
ax.set_title('100 epoches')
plt.show()

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
scatter = ax.scatter(latent_feature_450_x, latent_feature_450_y, c = cc , label = label)
legend1 = ax.legend(handles=[pop_0,pop_1,pop_2,pop_3,pop_4,pop_5,pop_6,pop_7,pop_8,pop_9],loc="lower left", title="Classes")
ax.add_artist(legend1)
ax.set_title('300 epoches')
plt.show()

#4
label_class = [0,1,2,3,4,5,6,7,8,9]
pred = [np.argmax(net1.forward(x)) for x,y in test_data]
real = [np.argmax(y) for x,y in test_data]
cnf_matrix = confusion_matrix(real, pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=label_class, normalize=False, title="confusion matrix")
plt.show()
