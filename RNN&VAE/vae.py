# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:44:13 2020

@author: user
"""

import numpay as np
import torch
from torchvision import datasets,transforms
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv_encode = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), #(64, 64, 3) -> (32, 32, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), #(32, 32, 64) -> (16, 16, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), #(16, 16, 128) -> (8, 8, 256)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1), #(8, 8, 128) -> (4, 4, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            )
        self.fc1 = nn.Linear(4*4*512, 128)
        self.fc2 = nn.Linear(4*4*512, 128)
    
        self.fc3 = nn.Linear(128, 4*4*512) 
        self.conv_decode = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1), #(4, 4, 512) -> (8, 8, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), #(8, 8, 256) -> (16, 16, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), #(16, 16, 128) -> (32, 32, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1), #(32, 32, 64) -> (64, 64, 3)
            nn.BatchNorm2d(3),
            nn.Sigmoid(),)
        
    def encoder(self, x):
        z = self.conv_encode(x)
        z = z.view(z.size(0), -1)
        
        mu = self.fc1(z)
        log_var = self.fc2(z)
        
        return mu, log_var
    
    def decoder(self, x, batch_size):
        z = self.fc3(x)
        z = z.view(batch_size, 512, 4, 4)
        out = self.conv_decode(z)
        return out
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        epslion = torch.randn_like(std)
        
        return mu + std * epslion
    
    def forward(self, x, batch_size):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z, batch_size)
        
        return x_reconst, mu, log_var
    

transform = transforms.Compose([transforms.Resize((64, 64)), 
                                transforms.ToTensor(),
                               ])
anime_set = datasets.ImageFolder('./data', transform)

BATCH_SIZE = 256
EPOCH = 100
lr = 0.0001
anime_loader = torch.utils.data.DataLoader(dataset=anime_set, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
net = VAE()
net = net.to(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

optimizer = torch.optim.Adam(net.parameters(), lr = lr)
#two three
# training
history_loss = []

for epoch in range(EPOCH):
    for step, (x, b_y) in enumerate(anime_loader):
        x = x.to(device)
        x_reconst, mu, log_var = net(x, x.size(0))
        
        # calculate loss
        reconst_loss = -F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_divergence = 0.5 * torch.sum( torch.pow(mu, 2) +torch.pow(log_var, 2) -torch.log(1e-8 + torch.pow(log_var, 2)) - 1).sum()
        ELBO = reconst_loss - kl_divergence
        loss = -ELBO

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(' EPOCH: %d, step: %d【Training】Loss: %.3f  ' % (epoch, step , loss.item()))
    torch.save(net.state_dict(), 'model/VAE_model.pth')
    if(epoch % 10 == 0):
        with torch.no_grad():
            sample = torch.randn(BATCH_SIZE, 128).to(device)
            imgs = net.decoder(sample, batch_size=BATCH_SIZE).cpu()
            fig = plt.gcf()
            fig.set_size_inches((12, 5))
            fig.suptitle('Epoch:'+ str(epoch), fontsize=15, color='r')
            for i in range(10):
                ax = plt.subplot(2, 5, i+1)
                ax.imshow(imgs[i].permute(1, 2, 0))
        plt.savefig('training(epoch:' + str(epoch) + ').png')               
    history_loss.append(loss.item())

plt.plot(history_loss)
plt.title('loss cruve', color='r')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#four
fig = plt.gcf()
fig.set_size_inches((12, 10))
fig.suptitle('reconstruction vs origin images', fontsize=15, color='r')
pos_idx = 0
for i in range(20):
    if(pos_idx == 20):
        break
    ax = plt.subplot(5, 4, pos_idx+1)
    ax.imshow(x[i].permute(1, 2, 0))
    
    ax = plt.subplot(5, 4, pos_idx+2)
    ax.imshow(x_reconst[i].permute(1, 2, 0))
    pos_idx += 2
    
#five
def interpolation(pro, model, img1, img2):    
    with torch.no_grad():    
        img1 = img1.to(device)
        mu_1, _ = model.encoder(torch.unsqueeze(img1,0))

        img2 = img2.to(device)
        mu_2, _ = model.encoder(torch.unsqueeze(img2,0))
        inter_latent = pro* mu_1 + (1- pro) * mu_2
        inter_image = model.decoder(inter_latent,1)
        inter_image = inter_image.cpu()

        return inter_image

pro_range=np.linspace(0,1,10)

fig, axs = plt.subplots(1,10, figsize=(20, 10))
axs = axs.ravel()

for ind,l in enumerate(pro_range):
    inter_image=interpolation(float(l), net, anime_set[0][0], anime_set[1][0])  
    inter_image = inter_image.clamp(0, 1)    
    image = inter_image.numpy()
    axs[ind].axis('off')
    axs[ind].imshow(image[0].transpose((1,2,0)))

plt.show() 