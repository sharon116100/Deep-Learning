from __future__ import print_function
#%matplotlib inline
import glob, os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import Compose, ToTensor
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64



# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
    
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Dropout(0.15),  # drop 50% of the neuron
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Dropout(0.15),  # drop 50% of the neuron
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Dropout(0.15),  # drop 50% of the neuron
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.Dropout(0.15),  # drop 50% of the neuron
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Dropout(0.15),  # drop 50% of the neuron
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

batch_size = 128
image_size = 64
num_epochs = 7
lr = 0.0002
beta1 = 0.5
workers = 0

def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
#    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fixed_noise = torch.randn(batch_size, nz, 1, 1)
    fake_label = 0
    loss_D_history = []
    loss_G_history = []
    D_update = []
    DG1_update = []
    DG2_update = []
    # Each epoch, we have to go through every data in dataset
    for epoch in range(5,num_epochs):
        # Each iteration, we will get a batch data for training
        for step, (x, b_y) in enumerate(dataloader):

            ############################
            """noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label = torch.full((batch_size,), fake_label, device=device)

            optimizer_d.zero_grad()
            real_x = x.to(device)
            b_y = b_y.to(device)

            data = torch.cat((fake, real_x), 0)
            new_label = torch.cat((label, b_y), 0)

            output = discriminator(data)
            loss_real = criterion(output, new_label)
            loss_real.backward()
            D_x = output.mean().item()
            D_G_z1 = output.mean().item()
            Loss_D = loss_real + loss_fake
            optimizer_d.step()"""
            ###########################

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            optimizer_d.zero_grad()
#            real_x = x.to(device)
            real_x = x
#            b_y = torch.full((batch_size,1), 1, device=device)
            b_y = torch.full((batch_size,1), 1)

            output = discriminator(real_x)
            loss_real = criterion(output, b_y)
            loss_real.backward()
            D_x = output.mean().item()

            # train with fake
#            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            noise = torch.randn(batch_size, nz, 1, 1)
            fake = generator(noise)
#            label = torch.full((batch_size,1), fake_label, device=device)
            label = torch.full((batch_size,1), fake_label)
            output = discriminator(fake.detach())
            loss_fake = criterion(output, label)
            loss_fake.backward()
            D_G_z1 = output.mean().item()
            Loss_D = loss_real + loss_fake
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            output = discriminator(fake)
            Loss_G = criterion(output, b_y)
            Loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            loss_D_history.append(Loss_D.item())
            loss_G_history.append(Loss_G.item())
            D_update.append(D_x)
            DG1_update.append(D_G_z1)
            DG2_update.append(D_G_z2)
            print('EPOCH:%d, [%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, step, len(dataloader),
                    Loss_D.item(), Loss_G.item(), D_x, D_G_z1, D_G_z2))
            if step %100 == 0:
                # do checkpointing
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),
                        './img_GAN/fake_samples_epoch_%d_%d.png' % (epoch,step),
                        normalize=True)
                torch.save(generator.state_dict(), './model/generator_epoch_%d_%d.pth' % (epoch, step))
                torch.save(discriminator.state_dict(), './model/discriminator_epoch_%d_%d.pth' % (epoch, step))
        
    plt.plot(loss_D_history, label = 'loss_D_history')
    plt.plot(loss_G_history, label = 'loss_G_history')
    plt.title('loss cruve', color='r')
    plt.xlabel('steps')
    plt.ylabel('Loss_G')
    plt.legend(loc='upper right')
    plt.savefig('./loss/loss_DG_%d_%d.png' % (epoch,step), format='png', transparent=True, dpi=300, pad_inches = 0)
    return loss_D_history, loss_G_history, D_update, DG1_update, DG2_update

     
            # Remember to save all things you need after all batches finished!!!
        
def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert('RGB')

def make_dataset(data_path):
    img_names = glob.glob(data_path+"/*.jpg")
    samples = []
    for img_name in img_names:
#        img_path = os.path.join(data_path, img_name)
        target = 1
        samples.append((img_name, target))
    return samples

class AlignData(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.samples = make_dataset(self.data_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return torch.Tensor(np.array(img)), torch.Tensor(target)



# Create the dataset by using ImageFolder(get extra point by using customized dataset)
# remember to preprocess the image by using functions in pytorch
transform = transforms.Compose([transforms.Resize((64, 64)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
dataset = AlignData('../img_align_celeba/img_align_celeba',transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                      drop_last=True,shuffle=True, num_workers=int(workers))



# Create the generator and the discriminator()
# Initialize them 
# Send them to your device
#generator = Generator(ngpu).to(device)
#generator = Generator(ngpu)
#discriminator = Discriminator(ngpu).to(device)
#discriminator = Discriminator(ngpu)

generator = Generator(ngpu).to(device)
generator.load_state_dict(torch.load('./model/generator_epoch_4_1500.pth'))
discriminator = Discriminator(ngpu).to(device)
discriminator.load_state_dict(torch.load('./model/discriminator_epoch_4_1500.pth'))
#noise = torch.randn(batch_size, nz, 1, 1, device=device)
#fake = generator(noise)
#plt.imshow((fake[0].cpu().detach().numpy().transpose((1,2,0))*128+127).astype(int))



# Setup optimizers for both G and D and setup criterion at the same time
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()
loss_D,loss_G,D,DG1,DG2 = train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs)
