import numpy as np
import torch as tr
import torch.nn as nn
from torch import distributions
from torch.nn.parameter import Parameter

import integrators as i
import update as u

import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import sys

import time


import matplotlib.pyplot as plt
import matplotlib as mpl
#bayesian analysis
import pymc as pm
import arviz as az


import os
os.environ["PATH"] = "/sciclone/home/yacahuanamedra/texlive/bin/x86_64-linux:" + os.environ["PATH"]

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')
import pickle


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}
mpl.rc('font', **font)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}
mpl.rc('font', **font)



import Gamma_error as gm
import integrators as integ
import njl as njl
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run NJL HMC sampling.")
    parser.add_argument("--L", type=int, default=16, help="Lattice size L (square lattice).")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size.")
    parser.add_argument("--mass", type=float, default=0.0, help="Fermion mass.")
    parser.add_argument("--g", type=float, default=0.4, help="Coupling g.")
    parser.add_argument("--device", type=str, default="cuda", help='Torch device (e.g., "cuda" or "cpu").')
    parser.add_argument("--Nwarm", type=int, default=100, help="Number of warmup trajectories.")
    parser.add_argument("--Nmeas", type=int, default=300, help="Number of measurements.")
    parser.add_argument("--Nskip", type=int, default=1, help="Number of trajectories between measurements.")
    parser.add_argument("--hmcsteps", type=int, default=14, help="Number of HMC integrator steps.")
    return parser.parse_args(argv)


args = parse_args(sys.argv[1:])





L = args.L
batch_size = args.batch_size

mass = args.mass
g = args.g
device = args.device


m_njl = njl.field(V=(L, L), g=g, m=mass, Nf=2, batch_size=batch_size, dtype=tr.float64, device=device)
Nwarm = args.Nwarm
Nmeas = args.Nmeas
Nskip = args.Nskip
hmcsteps = args.hmcsteps


sigma = m_njl.coldStart()
print(sigma.shape)
current_sigma=m_njl.generate_phi(sigma)

integr = integ.minnorm2(m_njl.force, m_njl.evolveQ, hmcsteps, 1.0)
samp = u.hmc(m_njl, integr, verbose=True)

sigma= samp.evolve(sigma,Nwarm)

tic=time.perf_counter()
Vol=m_njl.Vol
lat=[sigma.shape[1], sigma.shape[2]]
toc=time.perf_counter()

print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory", flush=True)

lC2p = []
lchi_m = []
E = []
av_sig = []
history_C_pi = []
history_C_sig=[]

phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]),dtype=m_njl.dtype,device=m_njl.device)
for k in range(Nmeas):
    ttE = m_njl.action(sigma)/Vol
    E.append(ttE)
    av_sigma = tr.mean(sigma.view(m_njl.Bs,Vol),axis=1)
    av_sig.append(av_sigma)
    chi_m = av_sigma*av_sigma*Vol
    p1_av_sig = tr.mean(sigma.view(m_njl.Bs,Vol)*phase.view(1,Vol),axis=1)

    #c_pi, c_sig = m_njl.measure_correlators(sigma)
    #history_C_pi.append(c_pi)

    results = m_njl.measure_momentum_correlators(sigma, k_list=[0,1,2,3,4,5],smearing="momentum",n_steps=30)
    history_C_pi.append(results['pion'])
    history_C_sig.append(results['sigma'])

    #C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
    if k%(Nmeas//100)==0:
        print("k= ",k,"(av_sigma,chi_m, E) ", av_sigma.cpu().mean().numpy(),chi_m.cpu().mean().numpy(),ttE.cpu().mean().numpy(),flush=True)
        #print("len(C2p): ", len(C2p))
    #lC2p.append(C2p)
    lchi_m.append(chi_m)
    ## HMC update but also V cycle
    sigma = samp.evolve(sigma,Nskip)


from collections import defaultdict
cpt = defaultdict(list)
cpt_sig=defaultdict(list)
for r in history_C_pi:
    for key, value in r.items():
        cpt[key].append(value)
for r in history_C_sig:
    for key, value in r.items():
        cpt_sig[key].append(value)
#tr.stack(cpt[5])



# Constantes de tu simulación
#L = 16 # O el tamaño de tu red
half_L = L // 2
batch_size = 50 

all_E = []
error_all_E = []

for P_n in range(6):
    Es = []
    errorEs = []
    
    # Ahora puedes ir desde 0 hasta half_L sin miedo a salirte de la red
    for time in range(L//2): 
        
        # --- EL TRUCO DE LA PERIODICIDAD (%) ---
        t_prev = (time - 1) % L
        t_curr = time % L
        t_next = (time + 1) % L
        
        # Extraemos con los índices que dan la vuelta a la red periódica
        C_tm1 = tr.vstack(cpt[P_n])[:, t_prev].reshape(-1, batch_size).to("cpu")
        C_t   = tr.vstack(cpt[P_n])[:, t_curr].reshape(-1, batch_size).to("cpu")
        C_tp1 = tr.vstack(cpt[P_n])[:, t_next].reshape(-1, batch_size).to("cpu")

        replicas = []
        for b in range(batch_size):
            rep = tr.stack([C_tm1[:, b], C_t[:, b], C_tp1[:, b]], dim=1) 
            replicas.append(rep)

        # Observable: F = arccosh( (C(t-1) + C(t+1)) / (2*C(t)) )
        E_eff = lambda A: tr.acosh((A[0] + A[2]) / (2.0 * A[1]))

        try:
            results = gm.gamma_method_with_replicas(replicas, E_eff)
            Es.append(results['value'])
            errorEs.append(results['dvalue'])

            print(f"P_n = {P_n}, time = {time}")
            # print results
            print(f"F = {results['value']:.6f} ± {results['dvalue']:.6f} (±{results['ddvalue']:.6f})")
            print(f"tau_int = {results['tau_int']:.3f} ± {results['dtau_int']:.3f}")
            print(f"W_opt = {results['W_opt']}, Q = {results['Q']}")
            
        except Exception as e:
            Es.append(np.nan)
            errorEs.append(np.nan)
            
    all_E.append(Es)
    error_all_E.append(errorEs)


results_av = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(av_sig).T.unsqueeze(2).to("cpu")), lambda A: A[0], max_lag=2000)

results_lchi = gm.gamma_method_with_replicas(gm.split_first_dim_to_list(tr.stack(lchi_m).T.unsqueeze(2).to("cpu")), lambda A: A[0], max_lag=2000)   

print("Gamma results for average phi:")
print(f"F = {results_av['value']:.6f} ± {results_av['dvalue']:.6f} (±{results_av['ddvalue']:.6f})")
print(f"tau_int = {results_av['tau_int']:.3f} ± {results_av['dtau_int']:.3f}")
print(f"W_opt = {results_av['W_opt']}, Q = {results_av['Q']}")
print("Gamma results for susceptibility:")
print(f"F = {results_lchi['value']:.6f} ± {results_lchi['dvalue']:.6f} (±{results_lchi['ddvalue']:.6f})")
print(f"tau_int = {results_lchi['tau_int']:.3f} ± {results_lchi['dtau_int']:.3f}")
print(f"W_opt = {results_lchi['W_opt']}, Q = {results_lchi['Q']}")


# ---------------------------------------------------------
# 1. Data Selection
# ---------------------------------------------------------
P_n = 0    # Momentum index (0 for p=0, 1 for p=1)
t_min = 3  # Start at t=1 to avoid UV contamination at t=0
t_max = L//2  # End before statistical noise dominates

# Extract data (assuming all_E and error_all_E are previously defined)
t_data = np.arange(len(all_E[P_n]))
E_data = np.array(all_E[P_n], dtype=float)
E_err  = np.array(error_all_E[P_n], dtype=float)

# Filter valid data within the specified time range
mask = (t_data >= t_min) & (t_data <= t_max) & ~np.isnan(E_data)
t_fit = t_data[mask]
y_fit = E_data[mask]
err_fit = E_err[mask]

# ---------------------------------------------------------
# 2. PyMC Bayesian Model
# ---------------------------------------------------------
with pm.Model() as exponential_fit:
    
    # Priors Informativos para guiar al MCMC
    
    # E0: Sabemos visualmente que el plateau está alrededor del último punto.
    # Un sigma de 0.05 es suficiente para que busque en esa zona.
    E0 = pm.Normal('E0', mu=y_fit[-1], sigma=0.05)
    
    # A: La amplitud es aproximadamente la diferencia entre el primer pico y el plateau.
    A_guess = y_fit[0] - y_fit[-1]
    A = pm.TruncatedNormal('A', mu=A_guess, sigma=0.2, lower=0.0)
    
    # dE: Sabemos que cae rapidísimo (casi desaparece en un solo paso temporal).
    # Por lo tanto, dE debe ser grande (~1.0 o más).
    dE = pm.TruncatedNormal('dE', mu=1.0, sigma=0.5, lower=0.0)
    
    # Curva Teórica
    mu = E0 + A * pm.math.exp(-dE * t_fit)
    
    # Likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=err_fit, observed=y_fit)
    
    # Muestreo (Aumentamos target_accept para evitar divergencias)
    print("Iniciando MCMC...")
    trace = pm.sample(draws=2000, tune=1500, chains=4, target_accept=0.99, random_seed=42)
    
    # Posterior predictive sampling
    ppc = pm.sample_posterior_predictive(trace)

# ---------------------------------------------------------
# 4. Results and Plotting
# ---------------------------------------------------------
summary = az.summary(trace, round_to=4)
print("\n--- Fit Results ---")
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

E0_mean = summary.loc['E0', 'mean']
A_mean  = summary.loc['A', 'mean']
dE_mean = summary.loc['dE', 'mean']
dE0 = summary.loc['E0', 'sd']
dA = summary.loc['A', 'sd']
ddE = summary.loc['dE', 'sd']

#save results E0, dE0 and g value for several g values
output_dir = "njldata_hmc"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f"NJL_results_L{L}_mass{mass}.txt")

with open(output_filename, "a") as f:
    if os.path.getsize(output_filename) == 0:    
        f.write("#g E0_mean E0_sd A_mean dE_mean hmcsteps dA ddE <sig> sig_sd <chi_m> chi_m_sd\n")
    f.write(f"{g} {E0_mean} {dE0} {A_mean} {dE_mean} {hmcsteps} {dA} {ddE} {results_av['value']} {results_av['dvalue']} {results_lchi['value']} {results_lchi['dvalue']}\n")
