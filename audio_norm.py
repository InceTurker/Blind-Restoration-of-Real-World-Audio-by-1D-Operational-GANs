import scipy.io as sio
import numpy as np

tt=sio.loadmat("tmats/clear_sig.mat")["clear_sig"]
maxtt=tt.max(axis=1)
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
clear_sig=normtt
sio.savemat("trainmats/clear_sig.mat", {'clear_sig':clear_sig})


tt=sio.loadmat("tmats/noisy_sig.mat")["noisy_sig"]
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
noisy_sig=normtt
sio.savemat("trainmats/noisy_sig.mat", {'noisy_sig':noisy_sig})

sio.savemat("train_max.mat", {'train_max':maxtt})

# ================================================================


tt=sio.loadmat("vmats/clear_sig.mat")["clear_sig"]
maxtt=tt.max(axis=1)
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
clear_sig=normtt
sio.savemat("validmats/clear_sig.mat", {'clear_sig':clear_sig})


tt=sio.loadmat("vmats/noisy_sig.mat")["noisy_sig"]
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
noisy_sig=normtt
sio.savemat("validmats/noisy_sig.mat", {'noisy_sig':noisy_sig})

sio.savemat("valid_max.mat", {'valid_max':maxtt})

# ================================================================


tt=sio.loadmat("temats/clear_sig.mat")["clear_sig"]
maxtt=tt.max(axis=1)
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
clear_sig=normtt
sio.savemat("testmats/clear_sig.mat", {'clear_sig':clear_sig})


tt=sio.loadmat("temats/noisy_sig.mat")["noisy_sig"]
normtt=np.zeros((len(maxtt),32000))
for i in range(len(maxtt)):
    normtt[i,:]=tt[i,:]/maxtt[i]
    
noisy_sig=normtt
sio.savemat("testmats/noisy_sig.mat", {'noisy_sig':noisy_sig})

sio.savemat("test_max.mat", {'test_max':maxtt})

