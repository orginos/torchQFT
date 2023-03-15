#!/usr/local/bin/python3
import numpy as np
import torch as tr
import phi4 as s
import integrators as i
import update as u

import matplotlib.pyplot as plt

device = "cuda" if tr.cuda.is_available() else "cpu"
print(f"Using {device} device")
    
def jackknife(d):
    # d is the list containing data
    # NOTE: it works with python lists not numpy arrays
    #
    N=len(d) -1 
    ss = sum(d)
    r=[]
    for n in range(len(d)):
        r.append((ss-d[n])/N)

    return r

def average(d):
    m = np.mean(d)
    e = np.std(d)/np.sqrt(len(d)-1)
    return m,e

def correlation_length(L,ChiM,C2p):
     return 1/(2*np.sin(np.pi/L))*np.sqrt(ChiM/C2p -1)
    
lat = [256,256]
lam = 0.5
mas = -0.205
Nwarm = 1000
Nmeas = 1000
Nskip = 10
Vol = np.prod(lat)

sg = s.phi4(lat,lam,mas,batch_size=4,device=device)

phi = sg.hotStart()
mn2 = i.minnorm2(sg.force,sg.evolveQ,7,1.0)
 
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))

hmc = u.hmc(T=sg,I=mn2,verbose=True)

phi = hmc.evolve(phi,Nwarm)

for k in range(phi.shape[0]):
    plt.imshow(phi[k,:,:], cmap='hot', interpolation='nearest')
    plt.show()
    

#save the last field configuration
tr.save(phi,'phi4_'+str(lat[0])+"_"+str(lat[1])+"_m"+str(mas)+"_l"+str(lam)+".pt")
