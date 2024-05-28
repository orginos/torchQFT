#!/usr/local/bin/python3
import time
import numpy as np
import scipy
import torch as tr
import phi4 as s
import integrators as i
import update as u

import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if tr.cuda.is_available() else "cpu"
if(device=="cpu"):
    device = "mps" if tr.backends.mps.is_available() else "cpu"
# OK I will always use CPU for now
device = "cpu"
device = tr.device(device)
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

L=16
lat = [L,L]
lam = 1.0
mas = -0.5
Nwarm = 5000
Nmeas = 1000
Nskip = 100
Vol = np.prod(lat)
Bs = 128
V=Vol

sg = s.phi4(lat,lam,mas,device=device,batch_size=Bs)

phi = sg.hotStart()
mn2 = i.minnorm2(sg.force,sg.evolveQ,7,1.0)
 
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))


hmc = u.hmc(T=sg,I=mn2,verbose=False)

tic=time.perf_counter()
phi = hmc.evolve(phi,Nwarm)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")

tic=time.perf_counter()
fx = tr.fft.fft2(phi)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e6:0.4f} micro-seconds per fft2")

pbar = tqdm(range(Nmeas))
c2p = tr.zeros([Nmeas,L,L],dtype=tr.float)
c2p_norw = tr.zeros([Nmeas,L,L],dtype=tr.float)
for k in pbar: 
     x=hmc.evolve(phi,Nskip)
     fx = tr.fft.fft2(x).detach()
     c2p[k,:,:] = (tr.real(fx*tr.conj(fx))).mean(dim=0).detach()

m_c2p=c2p.mean(dim=0)
e_c2p=c2p.std(dim=0)/np.sqrt(Nmeas-1)
ic2p= 1.0/m_c2p
ec2p= ic2p*(e_c2p/m_c2p)
plt.errorbar(np.arange(L),ic2p[:,0],ec2p[:,0],marker='.')
plt.show()


p2 = tr.zeros_like(m_c2p)
for x in range(L):
    for y in range(L):
        p2[x,y] = 0.5*(1-np.cos(2*np.pi*x/L) +  1-np.cos(2*np.pi*y/L))


res=scipy.stats.linregress(p2.view(V),ic2p.view(V))
x = np.linspace(0,2.0,100)
y = res.slope*x + res.intercept

xi = np.sqrt(res.slope/res.intercept)
e_xi = 1.0/2.0*(res.stderr/res.slope + res.intercept_stderr/res.intercept)*xi
print("The correlation length: ",xi,"+/-",e_xi)


slp2,indx = p2.view(V).sort()
sc2p = ic2p.view(V)[indx]

cut=8
res=scipy.stats.linregress(slp2[1:cut],sc2p[1:cut])
yy = res.slope*x + res.intercept

xi = np.sqrt(res.slope/res.intercept)
e_xi = 1.0/2.0*(res.stderr/res.slope + res.intercept_stderr/res.intercept)*xi
print(res)
print("The correlation length: ",xi,"+/-",e_xi)

plt.plot(slp2,sc2p,'.',x,y,x,yy)
plt.show()

cut2=10
plt.plot(slp2[:cut],sc2p[:cut],'.',x[:cut2],y[:cut2],x[:cut2],yy[:cut2])
plt.show()


#save the last field configuration
tr.save(phi,'phi4_'+str(lat[0])+"_"+str(lat[1])+"_m"+str(mas)+"_l"+str(lam)+"_b"+str(Bs)+".pt")
