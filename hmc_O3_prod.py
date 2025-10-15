#!/usr/local/bin/python
import time
from tqdm import tqdm
import argparse, re
from pathlib import Path
import numpy as np
import torch as tr
import O3 as s
import integrators as i
import update as u
from jackknife import jackknife, jackknife_over_streams
import json
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt

def dot3(a,b): return (a*b).sum(dim=1)

#def average(d):
#    m = np.mean(d,axis=0)
#    e = np.std(d,axis=0,ddof=1)/np.sqrt(len(d))
#    return m,e

# do full average of the tensor
def average(d):
    m = d.mean() 
    e = d.std(unbiased = True)/np.sqrt(d.numel())
    return m,e

def correlation_length(L,ChiM,C2p):
     return 1/(2*np.sin(np.pi/L))*tr.sqrt(ChiM/C2p -1)


# ----------------- Dir & filenames -----------------
def run_dir_for(lat, beta, base_dir=".") -> Path:
    return Path(base_dir) / f"o3_{lat[0]}_{lat[1]}_b{beta}"

def ckpt_path_for(lat, beta, batch_size, base_dir=".") -> Path:
    d = run_dir_for(lat, beta, base_dir)
    return d / f"o3_{lat[0]}_{lat[1]}_b{beta}_bs{batch_size}.pt"


def pos_int(s):
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return v

def pos_float(s):
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return v

def parse_L(s: str):
    """Parse L as one or two ints: '32' -> (32,32), '32x64'/'32,64'/'32 64' -> (32,64)."""
    parts = re.split(r"[xX, ]+", s.strip())
    parts = [p for p in parts if p]
    if len(parts) == 1:
        n = pos_int(parts[0])
        return (n, n)
    if len(parts) == 2:
        return (pos_int(parts[0]), pos_int(parts[1]))
    raise argparse.ArgumentTypeError("L must be like '32' or '32x64' or '32,64'")

# ----------------- Autocorrelation & IAT -----------------
def _autocorr_fft_1d(x: np.ndarray) -> np.ndarray:
    """Fast ACF via FFT, normalized so acf[0]=1."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.ones(1)
    # zero-pad to 2n for circular conv safety
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x: np.ndarray, c: float = 5.0, max_lag: int | None = None) -> float:
    """
    Sokal/emcee-style automatic windowing:
    tau = 1 + 2 * sum_{t=1..tmax} rho(t), stopping when t > c * tau or rho(t)<=0.
    """
    acf = _autocorr_fft_1d(x)
    n = acf.size
    if max_lag is None:
        max_lag = max(1, n // 2)
    tau = 1.0
    for t in range(1, max_lag):
        if acf[t] <= 0:
            break
        tau += 2.0 * acf[t]
        if t > c * tau:
            break
    return max(tau, 1.0)

# batched autocorrelation corrected average
def batched_average(f,tau=1): #N measurements over B batches
    N,B = f.shape
    err = tr.sqrt( (f.var(dim=0,unbiased=True)*2*tau/N).sum()   )/B
    return f.mean(),err

# ----------------- Packaging & saving -----------------
@dataclass
class Summary:
    Lx: int
    Ly: int
    beta: float
    Nwarm: int
    Nskip: int
    Nmeas: int
    Nmd: int
    batch_size: int

    xi: float | None
    xi_err: float | None
    chi_m: float
    chi_m_err: float
    c2p: float
    c2p_err: float
    chi_top: float
    chi_top_err: float

    tau_int_chi_m_mean: float
    tau_int_chi_m_std: float
    tau_int_c2p_mean: float
    tau_int_c2p_std: float
    tau_int_q_mean: float
    tau_int_q_std: float

def save_results(
    lat, beta, Nwarm, Nskip, Nmeas, Nmd, batch_size,
    chi_m_hist: tr.Tensor,   # [Nmeas, batch_size]
    c2p_hist  : tr.Tensor,   # [Nmeas, batch_size]
    q_hist: tr.Tensor,       # [Nmeas, batch_size]
    base_dir: str = "."):
    """
    Saves:
      run_dir/chi_m_history.pt  (float32) of shape [batch_size, Nmeas]
      run_dir/q_history.pt       (float32) of shape [batch_size, Nmeas]
      run_dir/summary.json
    Prints a short summary to stdout.
    """
    V = int(np.prod(lat))
    run_dir = run_dir_for(lat, beta, base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save histories
    tr.save(chi_m_hist, run_dir / "lchi_m_history.pt")
    tr.save(q_hist,      run_dir / "q_history.pt")

    # Per-stream IATs for quick decorrelation diagnostics
    def _per_stream_tau(hist_bt: tr.Tensor) -> np.ndarray:
        # hist_bt: [batch, Nmeas] -> list of tau for each stream
        taus = []
        for s in range(hist_bt.shape[0]):
            taus.append(integrated_autocorr_time(hist_bt[s].numpy()))
        return tr.tensor(taus,dtype=hist_bt.dtype) #np.asarray(taus, dtype=float)

    tau_chi_m = _per_stream_tau(chi_m_hist.T)
    tau_c2p   = _per_stream_tau(c2p_hist.T)
    tau_q     = _per_stream_tau(q_hist.T)

    chi_m, chi_m_err  = batched_average(chi_m_hist, tau_chi_m)
    c2p  , c2p_err    = batched_average(c2p_hist, tau_c2p)

    out = jackknife_over_streams(lambda x,x2: (x2.mean()-(x.mean())**2)/V, q_hist,q_hist**2 )
    chi_top = out["estimate"]
    chi_top_err = out["se"]

    out = jackknife_over_streams(lambda a,b: correlation_length(lat[0],a.mean(),b.mean()),chi_m_hist,c2p_hist)
    xi = out["estimate"]
    xi_err =  out["se"]

    
    summary = Summary(
        Lx=int(lat[0]), Ly=int(lat[1]), beta=float(beta),
        Nwarm=int(Nwarm), Nskip=int(Nskip), Nmeas=int(Nmeas), Nmd=int(Nmd), batch_size=int(batch_size),
        xi=float(xi), xi_err=float(xi_err),
        chi_m=float(chi_m), chi_m_err=float(chi_m_err),c2p=float(c2p), c2p_err=float(c2p_err),
        chi_top=float(chi_top), chi_top_err=float(chi_top_err),
        tau_int_chi_m_mean=float(tau_chi_m.mean()), tau_int_chi_m_std=float(tau_chi_m.std(unbiased=True)) if tau_chi_m.numel() >1 else 0.0,
        tau_int_c2p_mean=float(tau_c2p.mean()), tau_int_c2p_std=float(tau_c2p.std(unbiased=True)) if tau_c2p.numel() >1 else 0.0,
        tau_int_q_mean=float(tau_q.mean()),         tau_int_q_std=float(tau_q.std(unbiased=True)) if tau_q.numel() >1 else 0.0,
        
    )

    # Write JSON summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2)

    # Quick-glimpse printout
    print("\n=== Run summary ===")
    print(f"L={lat[0]}x{lat[1]}, beta={beta}, Nmeas={Nmeas}, batch={batch_size}")
    print(f"xi                    = {xi:.6f} ± {xi_err:.6f}  (jackknife)")
    print(f"chi_m                 = {chi_m:.6f} ± {chi_m_err:.6f}  (jackknife)")
    print(f"c2p                   = {c2p:.6f} ± {c2p_err:.6f}  (jackknife)")
    print(f"topological chi       = {chi_top:.6e} ± {chi_top_err:.6e}  (jackknife)")
    print(f"IAT(chi_m) per stream: mean={summary.tau_int_chi_m_mean:.2f} (±{summary.tau_int_chi_m_std:.2f})")
    print(f"IAT(c2p)   per stream: mean={summary.tau_int_c2p_mean:.2f} (±{summary.tau_int_c2p_std:.2f})")
    print(f"IAT(q)     per stream: mean={summary.tau_int_q_mean:.2f} (±{summary.tau_int_q_std:.2f})")
    print(f"Saved histories and summary to: {run_dir}")


###
# MAIN PROGRAM STARTS HERE
###

parser = argparse.ArgumentParser(description="HMC O(3) run configuration")

# 1) load toggle (default: try to load)
parser.add_argument("--load", dest="load", action="store_true", default=True,
                    help="Try to load checkpoint if it exists (default: on).")
parser.add_argument("--no-load", dest="load", action="store_false",
                    help="Disable loading even if a checkpoint exists.")

# 2) simulation parameters
parser.add_argument("--L", type=parse_L, default=(32, 32),
                    help="Lattice size: '32' or '32x64' or '32,64' (default: 32x32).")
parser.add_argument("--beta", type=pos_float, default=1.0, help="Inverse coupling β (default: 1.0).")
parser.add_argument("--Nwarm", type=pos_int, default=1000, help="Number of warmup sweeps (default: 1000).")
parser.add_argument("--Nskip", type=pos_int, default=10,   help="Sweeps between measurements (default: 10).")
parser.add_argument("--Nmeas", type=pos_int, default=100,  help="Number of measurements (default: 100).")
parser.add_argument("--Nmd",   type=pos_int, default=10,   help="MD steps per trajectory (default: 10).")
parser.add_argument("--batch-size", "--batch_size", dest="batch_size",
                    type=pos_int, default=64, help="Batch size (default: 64).")
parser.add_argument("--ckpt-path", dest="ckpt_path", default=".",  help="Path for checkpoints (default: ./)")


args = parser.parse_args()

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
#L = 24 ;  beta = 1.375 ; Nmd =3 ; Nskip = 10

# Unpack for convenience
lat = args.L              # tuple: (Lx, Ly)
beta = args.beta
Nwarm, Nskip, Nmeas = args.Nwarm, args.Nskip, args.Nmeas
Nmd = args.Nmd
batch_size = args.batch_size

device = "cuda" if tr.cuda.is_available() else "cpu"
if(device=="cpu"):
    device = "mps" if tr.backends.mps.is_available() else "cpu"
# OK I will always use CPU for now
device = "cpu"
device = tr.device(device)
print(f"Using {device} device")


sg = s.O3(lat,beta,batch_size)

# --- save configuration (final snapshot) and analysis artifacts ---
run_dir = run_dir_for(args.L, args.beta,base_dir=args.ckpt_path)
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_path_for(args.L, args.beta, args.batch_size,base_dir=args.ckpt_path)

if args.load and ckpt_path.is_file():
    try:
        phi = tr.load(ckpt_path, map_location="cpu")  # adjust map_location as needed
        print(f"Loaded checkpoint from {ckpt_path}")
        # If you saved a state_dict for a model: model.load_state_dict(data)
        # If you saved a tuple/dict (e.g., {'phi': phi, ...}), access accordingly: data['phi']
    except Exception as e:
        raise RuntimeError(f"Found file but failed to load: {ckpt_path}") from e
else:
    print(f"Skipping load or checkpoint not found: {ckpt_path}")
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

meas_C2p = tr.empty(args.Nmeas, args.batch_size, dtype=tr.float32)
meas_chi_m = tr.empty(args.Nmeas, args.batch_size, dtype=tr.float32)
meas_E =  tr.empty(args.Nmeas, args.batch_size, dtype=tr.float32)
meas_av_sig = tr.empty(args.Nmeas, args.batch_size, 3, dtype=tr.float32)
meas_q = tr.empty(args.Nmeas, args.batch_size, dtype=tr.float32)
pbar = tqdm(range(Nmeas))
for k in pbar:
    meas_E[k] = sg.action(phi)/sg.Vol
    
    av_sigma = phi.mean(dim=(2,3)) #tr.mean(phi.view(sg.Bs,sg.Vol),axis=1)
    meas_av_sig[k] = av_sigma
    
    meas_chi_m[k] = dot3(av_sigma,av_sigma)*sg.Vol  #np.dot(av_sigma,av_sigma)*G.Vol

    p1_av_sig = (phi*phase.view(1,1,*lat)).mean(dim=(2,3))
    meas_C2p[k] = tr.real(dot3(tr.conj(p1_av_sig), p1_av_sig) )*sg.Vol

    meas_q[k] = sg.Q(phi)
    phi = hmc.evolve(phi,Nskip)
    pbar.set_postfix({
        'chi_m': f"{meas_chi_m[:k+1,:].mean().item():.3f}",
        'c2p'  : f"{meas_C2p[:k+1,:].mean().item():.3f}",
        'Q'    : f"{meas_q[k,0].item():.3f}"
    })
   
print(f"Acceptance: {hmc.calc_Acceptance()}")
hmc.AcceptReject = []

m_sig = meas_av_sig.mean(dim=(0,1),keepdim=True)
e_sig = meas_av_sig.std(unbiased = True).expand(3)/np.sqrt(meas_av_sig.numel())
print("m_sig: ",m_sig.squeeze().numpy(),e_sig.numpy())

meas_chi_m_subtr =  meas_chi_m - (m_sig**2).sum(dim=2)*sg.Vol # subtract the mean
m_chi_m, e_chi_m = average(meas_chi_m_subtr) 
m_C2p, e_C2p     = average(meas_C2p)   
print("Chi_m: ",m_chi_m, e_chi_m)
print("C2p  : ",m_C2p, e_C2p)

avE,eE = average(meas_E)
print("E    : ", avE.item() ,'+/-',eE.item())

xi = correlation_length(lat[0],m_chi_m, m_C2p)
print("The correlation length is: ",xi)




C, C_jk_mean, C_err, bias, C_corr = jackknife(meas_chi_m_subtr.flatten(),
                                              meas_C2p.flatten(),
                                              F = lambda a,b: correlation_length(lat[0],a,b))

print(f"Correlation length = {C:.6f}")
print(f"  Jackknife mean   = {C_jk_mean:.6f}")
print(f"  Bias estimate    = {bias:.6e}")
print(f"  Bias-corrected   = {C_corr:.6f}")
print(f"  Jackknife error  = {C_err:.6f}")



mQ,eQ = average(meas_q)
Q2 =meas_q.var()
print("The topological charge is: ",mQ.numpy(),"+/-",eQ.numpy())
print("The topological succeptibility is: ",Q2.numpy()/sg.Vol )

q = meas_q.flatten()
C, C_jk_mean, C_err, bias, C_corr = jackknife(q,q**2, F = lambda x,x2: (x2-x**2)/sg.Vol)
print(f"Topological succeptibility = {C:.6f}")
print(f"  Jackknife mean           = {C_jk_mean:.6f}")
print(f"  Bias estimate            = {bias:.6e}")
print(f"  Bias-corrected           = {C_corr:.6f}")
print(f"  Jackknife error          = {C_err:.6f}")


#plt.plot(range(len(lchi_m)),lchi_m, range(len(lC2p)),lC2p)
#plt.show()

#plt.plot(range(len(q)),q)
#plt.show()
#save the last field configuration
ckpt_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
tr.save(phi, ckpt_path)  # or tr.save(phi, str(ckpt_path))
print(f"Saved to {ckpt_path}")

# Compute and save summaries (pass xi_hist if you recorded it)
save_results(
    lat=args.L, beta=args.beta, Nwarm=args.Nwarm, Nskip=args.Nskip,
    Nmeas=args.Nmeas, Nmd=args.Nmd, batch_size=args.batch_size,
    chi_m_hist=meas_chi_m_subtr,
    c2p_hist=meas_C2p,
    q_hist=meas_q,
    base_dir=args.ckpt_path)

