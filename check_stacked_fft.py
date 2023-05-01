import numpy as np
import scipy

import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as p
import integrators as i
import update as u

import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import sys
    
import time
from stacked_model import *


parser = argparse.ArgumentParser()
parser.add_argument('-f' , default='no-load')
parser.add_argument('-d' , type=int  , default=1   )
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.0 )
parser.add_argument('-b' , type=int  , default=128 )
parser.add_argument('-w' , type=int  , default=16  )
parser.add_argument('-nl', type=int  , default=1   )
parser.add_argument('-nm', type=int  , default=1000)


args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

device = "cuda" if tr.cuda.is_available() else "cpu"
print(f"Using {device} device")

depth = args.d
L=args.L
batch_size=args.b

V=L*L
lam =args.g
mass=args.m

o  = p.phi4([L,L],lam,mass,batch_size=batch_size)

#set up a prior
normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
prior= distributions.Independent(normal, 1)

width=args.w
Nlayers=args.nl
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior)
models = []

print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())
        
sm = SuperModel(models,target =o.action )

c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)

sm.load_state_dict(tr.load(file))
sm.eval()

print("starting model check")
validate(batch_size,'foo',sm)

x = sm.sample(batch_size)
plt.imshow(x.detach()[0,:,:], cmap='hot', interpolation='nearest')
plt.show()

tic=time.perf_counter()
fx = tr.fft.fft2(x)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e6:0.4f} micro-seconds per fft2")

Nmeas =args.nm
pbar = tqdm(range(Nmeas))
c2p = tr.zeros([Nmeas,L,L],dtype=tr.float)
for k in pbar: 
     x=sm.sample(batch_size)
     ds = sm.diff(x).detach()
     ds = ds - ds.mean()
     foo = tr.exp(-ds)
     w = foo/tr.mean(foo)
     fx = tr.fft.fft2(x).detach()
     c2p[k,:,:] = (tr.real(fx*tr.conj(fx))*w[:,None,None]).mean(dim=0).detach()

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
