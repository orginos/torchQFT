#!/usr/local/bin/python3
import time
import numpy as np
import torch as tr
import phi4 as s
import integrators as i
import update as u

import matplotlib.pyplot as plt

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

L=256
lat = [L,L]
# This set of params is very very close to critical.
lam = 0.5
mas = -0.205
#
Nwarm = 1000
Nmeas = 1000
Nskip = 10
batch_size = 32

Vol = np.prod(lat)


sg = s.phi4(lat,lam,mas,batch_size=batch_size,device=device)

phi = sg.hotStart()
mn2 = i.minnorm2(sg.force,sg.evolveQ,7,1.0)
 
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))


hmc = u.hmc(T=sg,I=mn2,verbose=False)

tic=time.perf_counter()
phi = hmc.evolve(phi,Nwarm)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")

lC2p = []
lchi_m = []
E = []
av_phi = []
phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]))
for k in range(Nmeas):
    ttE = sg.action(phi)/Vol
    E.extend(ttE)
    av_sigma = tr.mean(phi.view(sg.Bs,Vol),axis=1)
    av_phi.extend(av_sigma)
    chi_m = av_sigma*av_sigma*Vol
    p1_av_sig = tr.mean(phi.view(sg.Bs,Vol)*phase.view(1,Vol),axis=1)
    C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
    if(k%10==0):
        print("k= ",k,"(av_phi,chi_m, c2p, E) ", av_sigma.mean().numpy(),chi_m.mean().numpy(),C2p.mean().numpy(),ttE.mean().numpy())
    lC2p.extend(C2p)
    lchi_m.extend(chi_m)
    phi = hmc.evolve(phi,Nskip)


m_phi, e_phi = average(av_phi)
print("m_phi: ",m_phi,e_phi)

m_chi_m, e_chi_m = average(np.array(lchi_m) - (m_phi**2)*Vol)
m_C2p, e_C2p     = average(lC2p)
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)
avE,eE = average(E)
print("E = ", avE ,'+/-',eE)

xi = correlation_length(lat[0],m_chi_m, m_C2p)
print("The correlation length is: ",xi)
jphi   = jackknife(av_phi)
jchi_m = jackknife(lchi_m)- np.array(jphi)**2 * Vol
jC2p   = np.array(jackknife(lC2p)) 


j_xi = correlation_length(lat[0],jchi_m,jC2p)

m_xi,e_xi = average(j_xi)


print("The correlation length from jackknife is is: ",m_xi," +/- ", e_xi)

plt.plot(range(len(lchi_m)),lchi_m)
plt.show()
plt.plot(range(len(lC2p)),lC2p)
plt.show()
plt.plot(range(len(E)),E)
plt.show()
plt.plot(range(len(av_phi)),av_phi)
plt.show()


plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
plt.show()

#save the last field configuration
tr.save(phi,'phi4_'+str(lat[0])+"_"+str(lat[1])+"_m"+str(mas)+"_l"+str(lam)+".pt")
