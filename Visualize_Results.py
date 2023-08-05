import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def signalPower(x):
    return np.mean(x**2)

def SNR(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return 10*np.log10(powS/powN)


go=sio.loadmat("temats/noisy_sig.mat")
gan_outputs=go["noisy_sig"]

go=sio.loadmat("temats/clear_sig.mat")
clear_sig=go["clear_sig"]

# go=sio.loadmat("temats/noiseadded.mat")
go=sio.loadmat("C:/Research/AudioSupervised - final/temats/orig6/noisy_sig.mat")
noisy_sig=go["noisy_sig"]

i_snr_vals=[]
o_snr_vals=[]
snr_vals=[]

for i in range(len(gan_outputs)):
    i_snr_vals.append(SNR(clear_sig[i,:] ,noisy_sig[i,:]-clear_sig[i,:]))
    o_snr_vals.append(SNR(clear_sig[i,:] ,gan_outputs[i,:]-clear_sig[i,:]))

plt.figure()
plt.plot(i_snr_vals, label='input snr')
plt.plot(o_snr_vals, label='output snr')
plt.grid()
plt.legend()
print(np.mean(o_snr_vals))
print(np.mean(i_snr_vals))

plt.figure()

a=650
plt.subplot(311)
plt.plot(clear_sig[a,:])
plt.title("Clean Audio Signal")
plt.grid()

plt.subplot(312)
plt.plot(noisy_sig[a,:])
plt.title("Corrupted Audio Signal")
plt.grid()

plt.subplot(313)
plt.plot(gan_outputs[a,:])
plt.title("Reconstructed Operational GAN Output Signal")
plt.grid()

plt.figure()
import torch
import torchaudio

spectrogram = torchaudio.transforms.Spectrogram(n_fft=256,win_length=256,hop_length=128)
plt.subplot(311)

spre = spectrogram(torch.tensor(clear_sig[a,:]))
plt.imshow(spre.log10().detach().numpy())
plt.title("Clean Audio Spectrogram")
plt.subplot(312)

spre = spectrogram(torch.tensor(noisy_sig[a,:]))
plt.imshow(spre.log10().detach().numpy())
plt.title("Corrupted Audio Spectrogram")
plt.subplot(313)

spre = spectrogram(torch.tensor(gan_outputs[a,:]))
plt.imshow(spre.log10().detach().numpy())
plt.title("Reconstructed Operational GAN Output Spectrogram")