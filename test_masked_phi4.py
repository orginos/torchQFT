#!/usr/local/bin/python3
import numpy as np

import matplotlib.pyplot as plt
import torch as tr

#phi = tr.load('phi4_256_256_m-0.205_l1.0.pt')
phi = tr.load('phi4_256_256_m-0.205_l0.5.pt')
Nb = phi.shape[0]

#phi = tr.load('phi4_256_256_m-0.210_l1.0.pt')
nbatch = phi.shape[0]
lat = phi.shape[1:]
lat2 = [int(lat[0]/2), int(lat[1]/2)]

#phi = phi.squeeze()
X=np.arange(0,lat[0],2)
Y=np.arange(0,lat[1],2)

tt = tr.zeros_like(phi,dtype=tr.bool)
ttX = tt.clone()
ttY = tt.clone()
ttY[:,:,Y] = True
ttX[:,X,:] = True
mask = tr.logical_and(ttX,ttY)
fmask = tr.logical_not(mask)
#print(fmask)

phi2 = tr.reshape(tr.masked_select(phi,mask),(Nb,lat2[0],lat2[1]))
print(phi2.shape)

rphi = tr.zeros_like(phi)
rphi[mask]=phi2.view(Nb*lat2[0]*lat2[1])

iphi = tr.where(mask,rphi,phi)

print("The difference after interpolation is: ", tr.sum(tr.abs(iphi-phi)).item())
for k in range(Nb):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].pcolormesh(phi[k,:,:],cmap='hot')
    axs[0, 0].set_title('original')
    axs[0, 1].pcolormesh(phi2[k,:,:],cmap='hot')
    axs[0, 1].set_title('coarse')
    axs[1, 0].pcolormesh(rphi[k,:,:],cmap='hot')
    axs[1, 0].set_title('coarse,zeros->fine')
    axs[1, 1].pcolormesh(iphi[k,:,:],cmap='hot')
    axs[1, 1].set_title('coarse,fine->fine')
    fig.tight_layout()
    plt.show()
    

#    plt.figure(1)
#    plt.pcolormesh(phi[0,:,:])
#    plt.show()
#    
#    plt.figure(2)
#    plt.pcolormesh(phi2[0,:,:])
#    plt.show()
#    
#    plt.figure(3)
#    plt.pcolormesh(iphi[0,:,:])
#    plt.show()
    
