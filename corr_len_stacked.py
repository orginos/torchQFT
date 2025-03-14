import numpy as np
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

def kurtosis(x,dim):
    #print(x.std(dim).shape)
    mx = x.mean(dim).view(x.shape[0],1,1)
    sx = x.std(dim).view(x.shape[0],1,1)
    d = (x - mx)/sx
    #d = (x - x.mean(dim)*tr.ones_like(x))/(x.std(dim)
    return (d**4).mean(dim)-3

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


def reweight_list(l,w):
    return (np.array(l)*np.array(w)).tolist()

def tr_jackknife(d,func):
     # d is a torch tensor with the first dimension being a batch
     # function is evaluated in the data
     r =tr.zeros_like(d)
     for j in range(d.shape[0]):
          tt = tr.cat((d[:j],d[(j+1):]),0)
          #print(tt,tt.std())
          r[j]=func(tt).mean()

     return r

def compute_observables(av_phi,lC2p,lchi_m,E,mag,mag2,mag4):
    m_phi, e_phi = average(av_phi)
    print("m_phi: ",m_phi,e_phi)
    m_mag, e_mag = average(mag)
    
    m_chi_m, e_chi_m = average(np.array(lchi_m) - (m_phi**2)*V)
    m_C2p, e_C2p     = average(lC2p)
    print("Chi_m: ",m_chi_m, e_chi_m)
    print("C2p  : ",m_C2p, e_C2p)
    avE,eE = average(E)
    print("E = ", avE ,'+/-',eE)
    print("Magnetization : ", m_mag, '+/-',e_mag)
    
    xi = correlation_length(L,m_chi_m, m_C2p)
    jphi   = jackknife(av_phi)
    jchi_m = jackknife(lchi_m)- np.array(jphi)**2 * V
    jC2p   = np.array(jackknife(lC2p)) 
    jmag2 = np.array(jackknife(mag2))
    jmag4 = np.array(jackknife(mag4))
    
    j_xi = correlation_length(L,jchi_m,jC2p)
    
    m_xi,e_xi = average(j_xi)
    e_xi *= len(j_xi)

    print("The correlation length from jackknife is: ",m_xi," +/- ", e_xi)

    jU = jmag4/(jmag2)**2
    jB = 1 - jU/3

    m_U,e_U = average(jU)
    e_U *= len(jU)
    m_B,e_B = average(jB)
    e_B *= len(jB)
    print("U = ",m_U," +/- ",e_U)
    print("B = ",m_B," +/- ",e_B)
    return m_phi, e_phi, m_mag, e_mag, m_chi_m, e_chi_m,  m_C2p, e_C2p, avE,eE, m_xi,e_xi, m_U,e_U, m_B,e_B 

parser = argparse.ArgumentParser()
parser.add_argument('-f' , default='no-load')
parser.add_argument('-d' , type=int  , default=1   )
parser.add_argument('-L' , type=int  , default=16  )
parser.add_argument('-m' , type=float, default=-0.5)
parser.add_argument('-g' , type=float, default=1.0 )
parser.add_argument('-b' , type=int  , default=1024)
parser.add_argument('-w' , type=int  , default=16  )
parser.add_argument('-nl', type=int  , default=1   )
parser.add_argument('-sb' , type=int  , default=1    )
parser.add_argument('-nc' , type=int  , default=1    )
parser.add_argument('-fbj', type=bool , default=False)
parser.add_argument('-sbj', type=bool , default=False)
parser.add_argument('-dev', type=int  , default=-1)
parser.add_argument('-plt', type=bool  , default=False)

args = parser.parse_args()

file=args.f
if(args.f=="no-load"):
     load_flag=False
else:
    load_flag=True
    file=args.f

device = tr.device("cpu")
# Check that MPS is available
if not tr.backends.mps.is_available():
    if not tr.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
     if (args.dev >= 0 ):
          device = tr.device("mps")

# Check that CUDA is available
if not tr.cuda.is_available():
    if not tr.backends.cuda.is_built():
        print("CUDA not available because the current PyTorch install was not "
              "built with CUDA enabled.")
    else:
        print("CUDA not available or you do not have a GPU on this machine.")

else:
     if (args.dev >= 0 ):
          device = tr.device("cuda:"+str(args.dev))

print(f"Using {device} device")

depth = args.d
L=args.L
batch_size=args.b
Nconvs = args.nc

V=L*L
lam =args.g
mass=args.m

o  = p.phi4([L,L],lam,mass,batch_size=batch_size)
phi = o.hotStart()

#set up a prior
normal = distributions.Normal(tr.zeros(V,device=device),tr.ones(V,device=device))
prior= distributions.Independent(normal, 1)

width=args.w
Nlayers=args.nl

#non parity symmetric (default) bijector
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
if(args.fbj):
     bij_list = []
     for k in range(2*Nconvs):
          bij_list.append(m.FlowBijector(Nlayers=Nlayers,width=width))
     bij = m.BijectorFactory(bij_list).bij

#use parity symmetric bijector
if(args.sbj):
     print("Using parity symmetric bijector")
     bij = lambda: m.FlowBijectorParity(Nlayers=Nlayers,width=width)
     if(args.fbj):
          bij_list = []
          for k in range(2*Nconvs):
               bij_list.append(m.FlowBijectorParity(Nlayers=Nlayers,width=width))
          bij = m.BijectorFactory(bij_list).bij

mg = lambda : m.MGflow([L,L],bij,m.RGlayer("average"),prior,Nconvs=Nconvs)
models = []

print("Initializing ",depth," stages")
for d in range(depth):
    models.append(mg())
        
sm = SuperModel(models,target =o.action)

c=0
for tt in sm.parameters():
    #print(tt.shape)
    if tt.requires_grad==True :
        c+=tt.numel()
print("parameter count: ",c)

tag = str(L)+"_m"+str(mass)+"_l"+str(lam)+"_w_"+str(width)+"_l_"+str(Nlayers)+"_nc_"+str(Nconvs)+"_st_"+str(depth)
if(args.fbj):
     tag = tag + "_fbj"
if(args.sbj):
     tag = tag + "_sbj"
if(load_flag):
    sm.load_state_dict(tr.load(file))
    sm.eval()

sm.to(device)


print("starting model check")
x=sm.sample(batch_size)
diff = sm.diff(x).detach()
for b in range(1,args.sb):
    x=sm.sample(batch_size)
    diff = tr.cat((diff,sm.diff(x).detach()),0)
    
m_diff = diff.mean()     
diff -= m_diff

std =  diff.std().numpy()
e_std = tr_jackknife(diff,tr.std).std().numpy()*np.sqrt(diff.shape[0]-1)
print("max  action diff: ", tr.max(diff.abs()).numpy())
print("min  action diff: ", tr.min(diff.abs()).numpy())
print("mean action diff: ", m_diff.detach().numpy())
print("std  action diff: ",std,"+/-",e_std)
#compute the reweighting factor
foo = tr.exp(-diff)
#print(foo)
w = foo/tr.mean(foo)

rw = w.mean().numpy()
e_rw = w.std().numpy()
print("mean re-weighting factor: " , rw)
print("std  re-weighting factor: " , e_rw)

ESS = (foo.mean())**2/(foo*foo).mean()
print("ESS                     : " , ESS.numpy())

Nmeas = args.sb

lC2p = []
lchi_m = []
E = []
av_phi = []
rw = []
mag = []
mag2 = []
mag4 = []
phase=tr.tensor(np.exp(1j*np.indices(tuple((L,L)))[0]*2*np.pi/L))
for k in range(Nmeas):
    print(k)
    phi = sm.sample(batch_size)
    A = o.action(phi)
    diff = sm.diff(phi).detach()
    diff -= diff.mean()
    E.extend(A.detach().numpy()/V)
    foo = tr.exp(-diff)
    w = foo/tr.mean(foo)
    rw.extend(w)
    av = phi.mean(axis=(1,2)).detach()
    mag.extend(av.abs().numpy())
    mag2.extend((av**2).numpy())
    mag4.extend((av**4).numpy())
    #print(av)
    av_phi.extend(av.numpy())
    chi_m = av*av*V
    p1_av_sig = tr.mean(phi.view(o.Bs,V)*phase.view(1,V),axis=1)
    C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*V 
    lC2p.extend(C2p.detach().numpy())
    lchi_m.extend(chi_m.numpy())

print("RAW observables")
m_phi, e_phi, m_mag, e_mag, m_chi_m, e_chi_m,  m_C2p, e_C2p, avE,eE, m_xi,e_xi, m_U,e_U, m_B,e_B = compute_observables(av_phi,lC2p,lchi_m,E,mag,mag2,mag4)
print("-------")
OBS=[L,args.m,std,e_std,ESS.numpy(), m_mag,e_mag,  m_chi_m, e_chi_m, m_xi,e_xi, m_U,e_U ]
#fmt=len(OBS) * '{:.3f} '
fmt= 4 * '{:.3f} ' + '{:.3e} ' + 8 * '{:.3f} '
print("      L  m2    std   estd  ESS   Mag   eMag  ChiM  eChiM Xi   eXi  U      eU")
print("OBS: ",fmt.format(*OBS))
print("-------")

av_phi = reweight_list(av_phi,rw)
lC2p = reweight_list(lC2p,rw)
lchi_m = reweight_list(lchi_m,rw)
E = reweight_list(E,rw)
mag = reweight_list(mag,rw)
mag2 = reweight_list(mag2,rw)
mag4 = reweight_list(mag4,rw)

print("Reweighted observables")
m_phi, e_phi, m_mag, e_mag, m_chi_m, e_chi_m,  m_C2p, e_C2p, avE,eE, m_xi,e_xi, m_U,e_U, m_B,e_B = compute_observables(av_phi,lC2p,lchi_m,E,mag,mag2,mag4)
print("-------")
RW_OBS=[L,args.m,std,e_std,ESS.numpy(), m_mag,e_mag,  m_chi_m, e_chi_m, m_xi,e_xi, m_U,e_U ]
#fmt=len(RW_OBS) * '{:.3f} '
fmt= 4 * '{:.3f} ' + '{:.3e} ' + 8 * '{:.3f} '
print("         L  m2    std   estd  ESS   Mag   eMag  ChiM  eChiM Xi   eXi  U      eU")
print("RW_OBS: ",fmt.format(*RW_OBS))
print("-------")

if(args.plt):
    plt.hist(rw,bins=100)
    plt.show()
    plt.hist(np.log(rw),bins=100)
    plt.show()
    plt.plot(range(len(lchi_m)),lchi_m)
    plt.show()
    plt.plot(range(len(lC2p)),lC2p)
    plt.show()
    plt.plot(range(len(E)),E)
    plt.show()
    plt.plot(range(len(av_phi)),av_phi)
    plt.show()
    
    
    plt.imshow(phi.detach()[0,:,:], cmap='hot', interpolation='nearest')
    plt.show()
    


