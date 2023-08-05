import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
from GAN_Arch_details import Upsample,Downsample,CycleGAN_Unet_Generator,CycleGAN_Discriminator
import seaborn as sn
from scipy.stats import norm
import scipy.signal as sig
import copy
import scipy.io as sio
from torch.autograd import Variable
from pystoi import stoi
import torchaudio
from torch_stoi import NegSTOILoss

def signalPower(x):
    return np.mean(x**2)

def SNR(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return 10*np.log10(powS/powN)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tt=sio.loadmat("valid_max.mat")["valid_max"].transpose()
sample_rate = 16000
loss_func = NegSTOILoss(sample_rate=sample_rate)
data_dir = 'input/gan-getting-started'
batch_size = 8

dm = ECGDataModule(data_dir, batch_size, phase='test')
dm.prepare_data()
dataloader = dm.train_dataloader()

G = CycleGAN_Unet_Generator().cuda()
D = CycleGAN_Discriminator().cuda()

spectrogram = torchaudio.transforms.Spectrogram(n_fft=256,win_length=256,hop_length=128)

 
    
num_epoch = 2000
lr=0.0002
betas=(0.5, 0.999)
G_params = list(G.parameters())
D_params = list(D.parameters())
optimizer_g = torch.optim.Adam(G_params, lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(D_params, lr=lr, betas=betas)
criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
total_loss_d, total_loss_g = [], []
result = {}
E=0.000001 
for e in range(1,num_epoch):
    print("Epoch: "+str(e))
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    for input_img, real_img in (dataloader): 
      if(0):
          # check beats
          plt.subplot(211)
          plt.plot(input_img[0,0,:].cpu().detach())
          plt.title("Noisy Audio Signal/Clear Audio Signal")
          plt.subplot(212)
          plt.plot(real_img[0,0,:].cpu().detach())
      input_img=input_img.cuda()
      real_img=real_img.cuda()
      real_label = torch.ones(input_img.size()[0], 1, 1).cuda()
      fake_label = torch.zeros(input_img.size()[0],  1, 1).cuda()
      # Generator 
      fake_img = G(input_img).cuda()
      fake_img_ = fake_img.detach() # commonly using 
      out_fake = D(fake_img).cuda()
      loss_g_bce = criterion_mse(out_fake, real_label) # binaryCrossEntropy
      loss_g_mae = criterion_mae(fake_img, real_img) # MSELoss
      loss_g = loss_g_bce + 1*LAMBDA * loss_g_mae 
      total_loss_g.append(loss_g.item())
      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_g.backward()
      optimizer_g.step()
      out_real = D(real_img)
      loss_d_real = criterion_mse(out_real, real_label)
      out_fake = D(fake_img_)
      loss_d_fake = criterion_mse(out_fake, fake_label)
      loss_d = loss_d_real + loss_d_fake 
      total_loss_d.append(loss_d.item())
      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_d.backward()
      optimizer_d.step()
      loss_g, loss_d, fake_img=np.mean(total_loss_g), np.mean(total_loss_d), fake_img.detach().cpu()
      total_loss_d.append(loss_d)
      total_loss_g.append(loss_g)
    if e%10 == 0:
        data_dir = "validmats"
        batch_size = 100    
        dm2 = TECGDataModule(data_dir, batch_size, phase='test')
        dm2.prepare_data()
        dataloader2 = dm2.train_dataloader()
        base, style = next(iter(dataloader2))
        net = G
        net.eval()
        predicted = []
        predicted=pd.DataFrame(data=predicted)
        actual = []
        actual=pd.DataFrame(data=actual)
        ractual = []
        ractual=pd.DataFrame(data=ractual)
        m=sio.loadmat("vmats/noisy_sig.mat")["noisy_sig"].max(axis=1)
        m2=sio.loadmat("vmats/clear_sig.mat")["clear_sig"].max(axis=1)        
        with torch.no_grad():
          for base, style in (dataloader2):     
              output = net(base.cuda()).squeeze().cpu()
              ganoutput=output.detach().numpy()
              ganoutput=pd.DataFrame(data=ganoutput)
              predicted=pd.concat([predicted,ganoutput])
              ganacc=base.detach().numpy().squeeze()
              reall=style.detach().numpy().squeeze()
              reall=pd.DataFrame(data=reall)
              ganacc=pd.DataFrame(data=ganacc)
              actual=pd.concat([actual,ganacc])
              ractual=pd.concat([ractual,reall])
        gan_outputs=predicted.values.reshape(len(predicted)*32000,1)
        real_outputs=actual.values.reshape(len(actual)*32000,1)
        ractual=ractual.values.reshape(len(ractual)*32000,1)
        gan_outputs=gan_outputs[:len(gan_outputs)]
        real_outputs=real_outputs[:len(real_outputs)]
        ractual=ractual[:len(ractual)]
        gan_outputs1=gan_outputs.reshape(int(len(gan_outputs)/32000),32000)
        real_outputs1=real_outputs.reshape(int(len(real_outputs)/32000),32000)
        ractual1=ractual.reshape(int(len(ractual)/32000),32000)
        normgan_outputs1=np.zeros((len(tt),32000))
        normreal_outputs1=np.zeros((len(tt),32000))
        normractual1=np.zeros((len(tt),32000))
        for i in range(len(tt)):
            normgan_outputs1[i,:]=gan_outputs1[i,:]*tt[i,:]
            normreal_outputs1[i,:]=real_outputs1[i,:]*tt[i,:]
            normractual1[i,:]=ractual1[i,:]*tt[i,:]
        n=[]
        g1=[]
        for i in range(len(tt)):
            d1 = stoi(ractual1[i,:], real_outputs1[i,:], 16000, extended=False)
            d2 = stoi(ractual1[i,:], gan_outputs1[i,:], 16000, extended=False)
            n.append(d1)
            g1.append(d2)            
        print("Corrupted Input, STOI: " + str(round(np.mean(n),3)))
        print("GAN Output Stage 1, STOI: " + str(round(np.mean(g1),3)))
        gan_outputs1=normgan_outputs1
        real_outputs1=normreal_outputs1
        ractual1=normractual1
        i_snr_vals=[]
        o_snr_vals=[]
        snr_vals=[]
        
        for i in range(len(gan_outputs1)):
            i_snr_vals.append(SNR(ractual1[i,:] ,real_outputs1[i,:]-ractual1[i,:]))
            o_snr_vals.append(SNR(ractual1[i,:] ,gan_outputs1[i,:]-ractual1[i,:]))
        from random import randrange
        a=randrange(len(gan_outputs1))
        print("Epoch : "+str(e)+" Input SNR : "+ str(round(np.mean(i_snr_vals),2))+ " Output SNR : "+str(round(np.mean(o_snr_vals),2))+ " Gen loss : "+str(round(loss_g,2))+ "Corrupted Input, STOI: " + str(round(np.mean(n),3)*100)+"GAN Output, STOI: " + str(round(np.mean(g1),4)*100))
      
        torch.save(G.state_dict(), 'weights/model_weights_'+str(e)+'_.pth')
        print("model_saved")
        with open('sample.txt', 'a') as f:
            f.write("\n"+"Epoch : "+str(e)+" Input SNR : "+ str(np.mean(i_snr_vals))+ " Output SNR : "+str(np.mean(o_snr_vals))+ " Gen loss : "+str(round(loss_g,2))+  "Corrupted Input, STOI: " + str(round(np.mean(n),3)*100)+"GAN Output, STOI: " + str(round(np.mean(g1),4)*100)) 
        