#!/usr/local/bin/python
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch as tr
import O3 as s
import integrators as i
import update as u
from jackknife import jackknife

import matplotlib.pyplot as plt

def dot3(a,b): return (a*b).sum(dim=1)

def average(d):
    m = np.mean(d,axis=0)
    e = np.std(d,axis=0,ddof=1)/np.sqrt(len(d))
    return m,e

def correlation_length(L,ChiM,C2p):
     return 1/(2*np.sin(np.pi/L))*np.sqrt(ChiM/C2p -1)


def build_checkpoint_path(lat, beta, batch_size, base_dir="."):
    # Matches: 'o3_'+str(lat[0])+"_"+str(lat[1])+"_b"+str(beta)+"_bs"+str(batch_size)+".pt"
    return Path(base_dir) / f"o3_{lat[0]}_{lat[1]}_b{beta}_bs{batch_size}.pt"


device = "cuda" if tr.cuda.is_available() else "cpu"
if(device=="cpu"):
    device = "mps" if tr.backends.mps.is_available() else "cpu"
# OK I will always use CPU for now
device = "cpu"
device = tr.device(device)
print(f"Using {device} device")

 
# from https://arxiv.org/abs/1808.08129
#L = 24 ; beta = 1.263 ; Nmd =3 ; Nskip = 10
#L = 36 ; beta = 1.370 ; Nmd =3 ; Nskip = 10
#L = 54 ; beta = 1.458 ; Nmd =3 ; Nskip = 10
#L = 80 ; beta = 1.535 ; Nmd =6 ; Nskip = 10
#L = 120 ; beta = 1.607 ; Nmd =3 ; Nskip = 10
#L = 180 ; beta = 1.677 ; Nmd =3 ; Nskip = 10
#L = 270 ; beta = 1.743 ; Nmd =3 ; Nskip = 10
#L = 404 ; beta = 1.807 ; Nmd =6 ; Nskip = 10

#for my critical slowing down study:
#L =  8 ;  beta = 1.050 ; Nmd =3 ; Nskip = 10
#L = 16 ;  beta = 1.273 ; Nmd =3 ; Nskip = 10
#L = 24 ;  beta = 1.376 ; Nmd =3 ; Nskip = 10
L = 36 ; beta = 1.468 ; Nmd =3 ; Nskip = 10

lat = [L,L]  

Nwarm = 100
Nmeas = 2000

batch_size = 32


sg = s.O3(lat,beta,batch_size)

ckpt_path = build_checkpoint_path(lat, beta, batch_size)
if ckpt_path.is_file():
    try:
        phi = tr.load(ckpt_path, map_location="cpu")  # adjust map_location as needed
        print(f"Loaded checkpoint from {ckpt_path}")
        # If you saved a state_dict for a model: model.load_state_dict(data)
        # If you saved a tuple/dict (e.g., {'phi': phi, ...}), access accordingly: data['phi']
    except Exception as e:
        raise RuntimeError(f"Found file but failed to load: {ckpt_path}") from e
else:
    print(f"File not found: {ckpt_path}")
    print("***Hot start***")
    phi = sg.hotStart()
    
    


mn2 = i.minnorm2(sg.force,sg.evolveQ,Nmd,1.0)
 
print(phi.shape,tr.mean(phi),tr.std(phi))

hmc = u.hmc(T=sg,I=mn2,verbose=False)

tic=time.perf_counter()
phi = hmc.evolve(phi,Nwarm)
toc=time.perf_counter()
print(f"time {(toc - tic)*1.0e3/Nwarm:0.4f} seconds per HMC trajecrory")
print(f"Acceptance: {hmc.calc_Acceptance()}")
hmc.AcceptReject = []

phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]))

lC2p = []
lchi_m = []
E = []
av_phi = []
q = []
pbar = tqdm(range(Nmeas))
for k in pbar:
    ttE = sg.action(phi)/sg.Vol
    E.extend(ttE)
    av_sigma = phi.mean(dim=(2,3)) #tr.mean(phi.view(sg.Bs,sg.Vol),axis=1)
    av_phi.extend(av_sigma)
    
    chi_m = dot3(av_sigma,av_sigma)*sg.Vol  #np.dot(av_sigma,av_sigma)*G.Vol
    p1_av_sig = (phi*phase.view(1,1,*lat)).mean(dim=(2,3))
    #C2p = np.real(np.dot(np.conj(p1_av_sig),p1_av_sig))*G.Vol
    #C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
    C2p = tr.real(dot3(tr.conj(p1_av_sig), p1_av_sig) )*sg.Vol
    #if(k%10==0):
    #    print("k= ",k,"(chi_m, c2p) ", chi_m,C2p)
    lC2p.extend(C2p)
    lchi_m.extend(chi_m) 
    q.extend(sg.Q(phi))
    #print(" Q: ",q[-1])
    phi = hmc.evolve(phi,Nskip)
    pbar.set_postfix({
        'chi_m': f"{chi_m.mean().item():.3f}",
        'c2p'  : f"{C2p.mean().item():.3f}",
        'Q'    : f"{q[-1].item():.3f}"
    })

   
print(f"Acceptance: {hmc.calc_Acceptance()}")
hmc.AcceptReject = []

m_phi, e_phi = average(av_phi)
print("m_phi: ",m_phi,e_phi)

lchi_m = np.array(lchi_m) - (np.dot(m_phi,m_phi))*sg.Vol # subtract the mean
lC2p = np.array(lC2p) # convert to numpy array
m_chi_m, e_chi_m = average(lchi_m)
m_C2p, e_C2p     = average(lC2p)
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)

avE,eE = average(E)
print("E    : ", avE ,'+/-',eE)


xi = correlation_length(lat[0],m_chi_m, m_C2p)
print("The correlation length is: ",xi)




C, C_jk_mean, C_err, bias, C_corr = jackknife(lchi_m, lC2p, F = lambda a,b: correlation_length(lat[0],a,b))

print(f"Correlation length = {C:.6f}")
print(f"  Jackknife mean   = {C_jk_mean:.6f}")
print(f"  Bias estimate    = {bias:.6e}")
print(f"  Bias-corrected   = {C_corr:.6f}")
print(f"  Jackknife error  = {C_err:.6f}")


q = np.array(q)
mQ = np.mean(q)
eQ = np.std(q)
Q2 = np.var(q)
print("The topological charge is: ",mQ,"+/-",eQ)
print("The topological succeptibility is: ",Q2/sg.Vol )
C, C_jk_mean, C_err, bias, C_corr = jackknife(q,q**2, F = lambda x,x2: (x2-x**2)/sg.Vol)
print(f"Topological succeptibility = {C:.6f}")
print(f"  Jackknife mean           = {C_jk_mean:.6f}")
print(f"  Bias estimate            = {bias:.6e}")
print(f"  Bias-corrected           = {C_corr:.6f}")
print(f"  Jackknife error          = {C_err:.6f}")


plt.plot(range(len(lchi_m)),lchi_m, range(len(lC2p)),lC2p)
plt.show()

plt.plot(range(len(q)),q)
plt.show()
#save the last field configuration
ckpt_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
tr.save(phi, ckpt_path)  # or tr.save(phi, str(ckpt_path))
print(f"Saved to {ckpt_path}")

