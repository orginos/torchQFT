#!/usr/local/bin/python3
import torch as tr
device = "cuda" if tr.cuda.is_available() else "cpu"

import numpy as np
import phi4 as s
import integrators as i
import update as u

import matplotlib.pyplot as plt

def jackknife(d):
    # d is the list containing data
    # NOTE: it works with python lists not numpy arrays
    #
    r=[]
    for n in range(len(d)):
        jj = d[:n] + d[n+1:]
        r.append(np.mean(jj))

    return r

# jackknife estimate of the variance
def jackknife_var(d):
    # d is the list containing data
    # NOTE: it works with python lists not numpy arrays
    #
    r=[]
    for n in range(len(d)):
        jj = d[:n] + d[n+1:]
        r.append(np.var(jj))

    return r



lat = [32,32]
mass = 1.0
lam  = 1.0
batch_size = 32

Nwarm = 1000
Nmeas = 1000
Nskip = 10

sg = s.phi4(lat,lam,mass,batch_size = batch_size, device=device)

phi = sg.hotStart()

mn2 = i.minnorm2(sg.force,sg.evolveQ,3,1.0)
hmc = u.hmc(T=sg,I=mn2,verbose=False)

phi = hmc.evolve(phi,Nwarm)

phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]))

lC2p = []
lchi_m = []

for k in range(Nmeas):
    av_phi = tr.mean(phi.view(sg.Bs,sg.Vol),axis=1)
    chi_m = av_phi*av_phi*sg.Vol
    p1_av_sig = tr.mean(phi.view(sg.Bs,sg.Vol)*phase.view(1,sg.Vol),axis=1)
    C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*sg.Vol 
    print("k= ",k,"(chi_m, c2p) ", chi_m.tolist(),C2p.tolist())
    lC2p.extend(C2p.tolist())
    lchi_m.extend(chi_m.tolist())
    phi = hmc.evolve(phi,Nskip)

print("Acceptance rate: ",hmc.calc_Acceptance())
m_chi_m, e_chi_m = (np.mean(lchi_m),np.std(lchi_m)/np.sqrt(len(lchi_m)-1))
m_C2p, e_C2p     = (np.mean(lC2p),np.std(lC2p)/np.sqrt(len(lC2p)-1))
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)


xi = 1/(2*np.sin(np.pi/lat[0]))*np.sqrt(m_chi_m/m_C2p -1)
print("The correlation length is: ",xi)
jchi_m = jackknife(lchi_m)
jC2p   = jackknife(lC2p)

j_xi =  1/(2*np.sin(np.pi/lat[0]))*np.sqrt(np.array(jchi_m)/np.array(jC2p) -1)

m_xi = np.mean(j_xi)
e_xi = np.std(j_xi)*np.sqrt(len(j_xi)-1) 

print("The correlation length from jackknife is is: ",m_xi," +/- ", e_xi)


