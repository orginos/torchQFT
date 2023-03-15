#!/usr/local/bin/python3
import numpy as np

import matplotlib.pyplot as plt
import torch as tr
import phi4_rg_trans as rg

#phi = tr.load('phi4_256_256_m-0.205_l1.0.pt')
phi = tr.load('phi4_256_256_m-0.205_l0.5.pt')
Nb = phi.shape[0]

p = rg.phi4_masked_rg(phi.shape[1:], Nb)
phi2 = p.coarsen(phi)
rphi = p.fine_zeros(phi2)
iphi = p.fine_replace(phi2,phi)
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
    

