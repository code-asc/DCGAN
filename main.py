import os
import torch
import pickle as pkl
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from generator import Generator
from discriminator import Discriminator

def realLoss(D_out):
    b_size = D_out.size(0)
    labels = torch.ones(b_size)
    labels = labels.to(device)
    loss_fun = torch.nn.BCELoss()
    loss = loss_fun(D_out.squeeze(), labels)
    return loss

def fakeLoss(D_out):
    b_size = D_out.size(0)
    labels = torch.zeros(b_size)
    labels = labels.to(device)
    loss_fun = torch.nn.BCELoss()
    loss = loss_fun(D_out.squeeze(), labels)
    return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)


seed = np.random.randint(1, 1000)
torch.manual_seed(seed)

PRINT_INTERVAL = 50
#root = "/floyd/input/data/"
root = '/data/celebs/'
worker = 2
batch_size = 128
image_size = 64
ngf = 64
EPOCH = 10
lr = 0.0002
beta1 = 0.5

transform = transforms.Compose([transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = dset.ImageFolder(root=root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")



dis = Discriminator().to(device)
gen = Generator().to(device)

if(device.type == 'cuda'):
    dis = torch.nn.DataParallel(dis)
    gen = torch.nn.DataParallel(gen)

dis.apply(weights_init)
gen.apply(weights_init)

#criterion = toech.nn.BCELoss()
fixed_z = torch.randn(64, 100, 1,1, device=device)

optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

samples = []

for epoch in range(EPOCH):
    for i, data in enumerate(dataloader, 0):

        data_ = data[0].to(device)
        b_size = data_.size(0)

        dis.zero_grad()
        optimizerD.zero_grad()
        #========================================
        #   Training Discriminator
        #   Discriminator loss on real and fake images
        #========================================
        d_real_pred = dis(data_)
        d_real_loss = realLoss(d_real_pred)

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = gen(noise)

        d_fake_pred = dis(fake_images)
        d_fake_loss = fakeLoss(d_fake_pred)

        d_total_loss = d_real_loss + d_fake_loss
        d_total_loss.backward()
        optimizerD.step()

        #========================================
        #   Training Generator
        #========================================
        optimizerG.zero_grad()
        gen.zero_grad()
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = gen(noise)

        d_fake_pred = dis(fake_images)
        g_loss = realLoss(d_fake_pred)


        g_loss.backward()
        optimizerG.step()

        if i % PRINT_INTERVAL == 0:
            print('Epoch [{:5d}/{:5d}] | d_total_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, EPOCH, d_total_loss.item(), g_loss.item()))


    samples_z = gen(fixed_z)
    samples.append(samples_z)


with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

print('saved samples....')

torch.save(gen.state_dict(), './generator_state.mdl')
print('Model state saved....')
torch.save(gen, './generator_model.mdl')
print('Model saved....')
