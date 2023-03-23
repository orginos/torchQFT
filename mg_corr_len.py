import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter
import phi4_mg as m
import phi4 as s
import integrators as i
import update as u

import time
import matplotlib.pyplot as plt

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

L=16
V=L*L
Vol=V
batch_size=128
lam =1.0
mass= -0.2
Nmeas = 100

o  = s.phi4([L,L],lam,mass,batch_size=batch_size)

#print(o.action(phi))
#set up a prior
normal = distributions.Normal(tr.zeros(V),tr.ones(V))
prior= distributions.Independent(normal, 1)

width=256
Nlayers=1

file = "phi4_"+str(L)+"_m"+str(mass)+"_l"+str(lam)+"_nvp_w"+str(width)+"_n"+str(Nlayers)+".dict"

#file ='phi4_16_m-0.2_l1.0_nvp_w512.dict'

print("Reading from file: ",file)
bij = lambda: m.FlowBijector(Nlayers=Nlayers,width=width)
mg = m.MGflow([L,L],bij,m.RGlayer("average"),prior)

#print(mg)

mg.load_state_dict(tr.load(file))
mg.eval()

#print(mg)
c=0
for tt in mg.parameters():
    if tt.requires_grad==True :
        c+=tt.numel()
        
print("parameter count: ",c)

print("The model depth is: ", mg.depth)
#print("RG layers: ",mg.rg)
#print("Convolutional flow  layers: ",mg.cflow)


lC2p = []
lchi_m = []
E = []
av_phi = []
rw = []
phase=tr.tensor(np.exp(1j*np.indices(tuple((L,L)))[0]*2*np.pi/L))
#print(phase)
#print(np.indices(tuple((L,L))))

for k in range(Nmeas):
    print(k)
    phi = mg.sample(batch_size)
    A = o.action(phi)
    diff = (A+mg.log_prob(phi)).detach()
    diff -= diff.mean()
    E.extend(A.detach().numpy()/Vol)
    foo = tr.exp(-diff)
    w = foo/tr.mean(foo)
    rw.extend(w)
    av = phi.mean(axis=(1,2)).detach()
    #print(av)
    av_phi.extend(av.numpy())
    chi_m = av*av*Vol
    p1_av_sig = tr.mean(phi.view(o.Bs,Vol)*phase.view(1,Vol),axis=1)
    C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol 
    lC2p.extend(C2p.detach().numpy())
    lchi_m.extend(chi_m.numpy())


av_phi = reweight_list(av_phi,rw)
lC2p = reweight_list(lC2p,rw)
lchi_m = reweight_list(lchi_m,rw)
E = reweight_list(E,rw)

m_phi, e_phi = average(av_phi)
print("m_phi: ",m_phi,e_phi)

m_chi_m, e_chi_m = average(np.array(lchi_m) - (m_phi**2)*Vol)
m_C2p, e_C2p     = average(lC2p)
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)
avE,eE = average(E)
print("E = ", avE ,'+/-',eE)

xi = correlation_length(L,m_chi_m, m_C2p)
print("The correlation length is: ",xi)
jphi   = jackknife(av_phi)
jchi_m = jackknife(lchi_m)- np.array(jphi)**2 * Vol
jC2p   = np.array(jackknife(lC2p)) 


j_xi = correlation_length(L,jchi_m,jC2p)

m_xi,e_xi = average(j_xi)


print("The correlation length from jackknife is is: ",m_xi," +/- ", e_xi)

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
