#!/usr/local/bin/python3
import numpy as np

import matplotlib.pyplot as plt
import torch as tr

phi = tr.load('phi4_256_256_m-0.205_l1.0.pt')
#phi = tr.load('phi4_256_256_m-0.210_l1.0.pt')
nbatch = phi.shape[0]
lat = phi.shape[1:]
lat2 = [int(lat[0]/2), int(lat[1]/2)]

#phi = phi.squeeze()

X=np.arange(0,lat[0],2)
Y=np.arange(0,lat[1],2)


#print(indY)

fooX = np.repeat(X[:,np.newaxis],lat[1],axis=1)
fooX = np.repeat(fooX[np.newaxis,:,:],nbatch,axis=0)
indX = tr.tensor(fooX)
fooY = np.repeat(Y[np.newaxis,:],lat2[0],axis=0)
fooY = np.repeat(fooY[np.newaxis,:,:],nbatch,axis=0)
indY = tr.tensor(fooY)
print("indX,indY",indX.shape,indY.shape)

xF=tr.gather(phi,1,indX)
phi2=tr.gather(xF,2,indY)

# now interpolate back
oX = X + 1
oY = Y + 1
fooX = np.repeat(oX[:,np.newaxis],lat[1],axis=1)
fooX = np.repeat(fooX[np.newaxis,:,:],nbatch,axis=0)
oindX = tr.tensor(fooX)
fooY = np.repeat(oY[np.newaxis,:],lat2[0],axis=0)
fooY = np.repeat(fooY[np.newaxis,:,:],nbatch,axis=0)
oindY = tr.tensor(fooY)
print ("oindX,oindY", oindX.shape,oindY.shape)

syF = tr.zeros_like(xF).scatter_(2,indY,phi2).scatter_(2,oindY,phi2)
sF = tr.zeros_like(phi).scatter_(1,indX,syF).scatter_(1,oindX,syF)

plt.figure(1)
plt.pcolormesh(phi[0,:,:])
plt.show()

plt.figure(2)
plt.pcolormesh(phi2[0,:,:])
plt.show()

plt.figure(3)
plt.pcolormesh(sF[0,:,:])
plt.show()


