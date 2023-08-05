import scipy.io as sio
from pystoi import stoi
import numpy as np
clear_sig=sio.loadmat("C:/Research/AudioSupervised - finalv2/temats/clear_sig.mat")["clear_sig"]
noiseadded=sio.loadmat("C:/Research/AudioSupervised - finalv2/temats/orig/noisy_sig.mat")["noisy_sig"]
ganout1=sio.loadmat("C:/Research/AudioSupervised - finalv2/temats/noisy_sig.mat")["noisy_sig"]


n=[]
g1=[]
g2=[]
snrxx=[]

for i in range(830):
    print(i)
  
    Es = np.sum(clear_sig[i,:] ** 2)
    noise=noiseadded[i,:] - clear_sig[i,:]
    En = np.sum(noise ** 2)
    ratio=Es/En
    snrx=10*np.log10(ratio)
    snrxx.append(snrx)
    

print("GAN Output Stage 2, STOI: " + str(round(np.mean(snrxx),4)) + "  " + str(round(np.std(snrxx),4)))

