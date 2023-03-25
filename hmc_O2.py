#!/usr/local/bin/python3
import numpy as np
import torch as tr
import O2 as s
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
    
lat = [64,64]
beta = 1.09
Nwarm = 1000
Nmeas = 2000
Nskip = 10
Vol = np.prod(lat)

o = s.O2(lat,beta,device=device)

phi = o.hotStart()
mn2 = i.minnorm2(o.force,o.evolveQ,4,1.0)
 
print(phi.shape,Vol,tr.mean(phi),tr.std(phi))

hmc = u.hmc(T=o,I=mn2,verbose=False)

phi = hmc.evolve(phi,Nwarm)

# we will calculate the helicity modulus
# (see https://arxiv.org/pdf/1210.6116.pdf)
lY = []
lC2p = []
lchi_m = []
E = []
l_av_sig = []
phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]))
for k in range(Nmeas):
    lY.append(o.helicity_modulus(phi).numpy())
    E.append(o.action(phi).item()/Vol)
    sigma = tr.zeros(o.Bs,2,o.V[0],o.V[1])
    sigma[:,0,:,:] = tr.cos(phi).view(o.Bs,1,o.V[0],o.V[1])
    sigma[:,1,:,:] = tr.sin(phi).view(o.Bs,1,o.V[0],o.V[1])
    av_sigma = tr.mean(sigma.view(o.Bs,2,Vol),axis=2)
    l_av_sig.append(av_sigma.numpy())
    chi_m = tr.sum(av_sigma*av_sigma*Vol,dim=(1))
    p1_av_sig = tr.mean(sigma.view(o.Bs,o.N,o.Vol)*phase.view(1,1,o.Vol),axis=2)
    C2p = tr.sum(tr.real(tr.conj(p1_av_sig)*p1_av_sig),dim=1)*Vol 
    print("k= ",k,"(av_sig,chi_m, c2p, E, Y) ", av_sigma.tolist(),chi_m.tolist(),C2p.tolist(),E[-1].tolist(),lY[-1].item())
    lC2p.append(C2p.item())
    lchi_m.append(chi_m.item())
    phi = hmc.evolve(phi,Nskip)


#m_phi, e_phi = average(av_phi)
#print("m_phi: ",m_phi,e_phi)

print("Acceptance rate: ",hmc.calc_Acceptance())

#print(lC2p)

m_sig = (sum(l_av_sig)/len(l_av_sig))[0]
print("m_sig: ",m_sig)

#print(lY)
mY, eY = average(lY)
m_chi_m, e_chi_m = average(np.array(lchi_m))
m_chi_m -= np.sum(m_sig*m_sig)*o.Vol
m_C2p, e_C2p     = average(lC2p)
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)
avE,eE = average(E)
print("E = ", avE ,'+/-',eE)
print("Y = ", mY ,'+/-',eY)

xi = correlation_length(lat[0],m_chi_m, m_C2p)
print("The correlation length is: ",xi)
jsig   = jackknife(l_av_sig)
tt = np.array(jsig).reshape(len(jsig),2)
tt2 = np.sum(tt*tt,axis=1)*o.Vol
#print("check tt2: ",tt2)
jchi_m = jackknife(lchi_m) -tt2
foo = average(jchi_m)
print("Chi_m from jackknife: ",foo[0],foo[1]*np.sqrt(Nmeas))
jC2p   = np.array(jackknife(lC2p))
foo = average(jC2p)
print("C2p from jackknife: ",foo[0],foo[1]*np.sqrt(Nmeas))

j_xi = correlation_length(lat[0],jchi_m,jC2p)

m_xi,e_xi = average(j_xi)
e_xi*=np.sqrt(j_xi.shape[0])


print("The correlation length from jackknife is is: ",m_xi," +/- ", e_xi)

plt.plot(range(len(lchi_m)),lchi_m)
plt.show()
plt.plot(range(len(lC2p)),lC2p)
plt.show()
plt.plot(range(len(E)),E)
plt.show()
foo = np.array(lY).squeeze()
plt.plot(range(foo.shape[0]),foo)
plt.show()
#plt.plot(range(len(l_av_phi)),np.sum(av_sig*av_sig))
#plt.show()


plt.imshow(phi[0,:,:], cmap='hot', interpolation='nearest')
plt.show()

#save the last field configuration
tr.save(phi,'O2_'+str(lat[0])+"_"+str(lat[1])+"_b"+str(beta)+".pt")
