#!/usr/local/bin/python3
## JUST TESTING FUNCTIONALITY
## 
import numpy as np

import matplotlib.pyplot as plt
import torch as tr
import integrators as i
import update as u
import phi4 as p
import phi4_rg_trans as rg


lam = 0.1
mass = -0.204

lat = [64,64]
Nwarm = 1000
ll = min(lat)
latt_sizes = [lat]
for k in range(int(np.log(ll)/np.log(2))-1):
    print(k)
    latt_sizes.append([int(latt_sizes[-1][0]/2),int(latt_sizes[-1][1]/2)])
    print(latt_sizes)


# initialize the model
# the list of theories
# and list of rg transformations
p4 = []
trans = []
hmc = []
tmp_lam = lam
for ll in latt_sizes:
    p4.append(p.phi4(ll,tmp_lam,mass))
    trans.append(rg.phi4_rg_trans(ll))
    hmc.append(u.hmc(T = p4[-1],I=i.minnorm2(p4[-1].force,p4[-1].evolveQ,7,1.0)
,verbose=True))
    # blocking factor is 2
    # transform the couplings according to their engineering dimension
    # mass is dimensionless while lambda is length^2
    tmp_lam *= 4
    print(tmp_lam)


# here HMC is for testing purposes only 
def coarse_to_fine(p4,trans,hmc):
    foo = p4[len(latt_sizes)-1].hotStart()
    for k in range(len(latt_sizes)-1,-1,-1):
        print("Lattice size: ",latt_sizes[k])
        print("The coupling is: ", p4[k].lam)
        if(k == len(latt_sizes)-1):
            print("shape of current field", foo.shape)
            #### HMC NEEDS TO BE REPLACED BY RealNVP LAYERS
            foo = hmc[k].evolve(foo,Nwarm)
        else:
            boo = trans[k].noisy_refine(foo)
            print("shape of current field", boo.shape)
            ##### HMC NEEDS TO BE REPLACED BY RealNVP LAYERS
            foo = hmc[k].evolve(boo,Nwarm)
        
    return foo

# here HMC is for testing purposes only 
def fine_to_coarse(phi, p4,trans,hmc):
    foo = phi.clone()
    for k in range(len(latt_sizes)):
        foo = hmc[k].evolve(foo,Nwarm)
        foo = trans[k].coarsen(foo)
    return foo

phi = coarse_to_fine(p4,trans,hmc)

c_phi = fine_to_coarse(phi, p4,trans,hmc)
print("Dimensions of coarse field: ", c_phi.shape)

