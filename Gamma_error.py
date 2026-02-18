import math
import torch as tr
import numpy as np
import time
from scipy.stats import chi2 as chi2dist

"""
Created on Wed Nov 2 03:30:00 2025

@author: Yamil Cahuana

"""

def get_observables_hist(sg, hmc, phi, Nwarm, Nmeas, Nskip, pp="no"):
    """
    Measures observables during an HMC evolution.
    """
    tic = time.perf_counter()
    Vol = sg.Vol
    lat = [phi.shape[1], phi.shape[2]]
    toc = time.perf_counter()

    print(f"Time {(toc - tic)*1.0e6/Nwarm:0.4f} microseconds per HMC trajectory")

    lC2p = []
    lchi_m = []
    E = []
    av_phi = []
    phase = tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]), dtype=sg.dtype, device=sg.device)
    
    for k in range(Nmeas):
        ttE = sg.action(phi)/Vol
        E.append(ttE)
        av_sigma = tr.mean(phi.view(sg.Bs, Vol), axis=1)
        av_phi.append(av_sigma)
        chi_m = av_sigma*av_sigma*Vol
        p1_av_sig = tr.mean(phi.view(sg.Bs, Vol)*phase.view(1, Vol), axis=1)
        C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
        
        if(k % 100 == 0) and pp == "print":
            print(f"k={k} (av_phi, chi_m, c2p, E) {av_sigma.mean().item():.4f}, {chi_m.mean().item():.4f}, {C2p.mean().item():.4f}, {ttE.mean().item():.4f}")
            
        lC2p.append(C2p)
        lchi_m.append(chi_m)
        phi = hmc.evolve(phi, Nskip)

    return lC2p, lchi_m, E, av_phi, phi

def compute_gradient_autograd(f, mean_a):
    """
    Eq. (17): Compute gradient f_alpha = ∂f/∂A_alpha using autograd at the mean point. [cite: 103, 104]
    """
    mean_a = mean_a.clone().detach().requires_grad_(True)
    f_value = f(mean_a)
    f_value.backward()
    return mean_a.grad.detach()

def split_first_dim_to_list(tensor: tr.Tensor) -> list:
    """
    Given a tensor of shape [N, T, D], returns a list of N tensors of shape [T, D].

    Args:
        tensor (torch.Tensor): A 3D tensor with shape [N, T, D].

    Returns:
        list of torch.Tensor: List of N tensors each with shape [T, D].
    """
    return [tensor[i] for i in range(tensor.shape[0])]


def autocorrelation_proj_with_replicas(data_replicas, grad_f, max_lag):
    """
    Eq. (31): Implementation of the Gamma-method autocorrelation for multiple replicas. [cite: 159]
    Estimates the projected autocorrelation function within each replica to avoid boundary noise. [cite: 162, 163]
    """
    # Project each replica separately Eq. (37) [cite: 191]
    projected_reps = [rep @ grad_f for rep in data_replicas]
    
    N_total = sum(len(r) for r in projected_reps)
    R = len(projected_reps)
    
    # Global mean based on all measurements Eq. (7) [cite: 69]
    all_projected = tr.cat(projected_reps)
    mean_global = all_projected.mean()
    
    gamma = tr.zeros(max_lag)
    
    for t in range(max_lag):
        s_sum = 0.0
        # Correlate within each replicum and then average Eq. (31) 
        for rep in projected_reps:
            if len(rep) > t:
                d1 = rep[:len(rep)-t] - mean_global
                d2 = rep[t:] - mean_global
                s_sum += (d1 * d2).sum()
        
        # Proper normalization for replica averages [cite: 159, 161]
        gamma[t] = s_sum / (N_total - R * t)

    rho = gamma / gamma[0] # Normalized autocorrelation rho(t) [cite: 433]
    
    # Statistical error of the autocorrelation function [cite: 434]
    rho_err = tr.zeros(max_lag)
    for t in range(max_lag):
        # Approximation based on the Madras-Sokal formula [cite: 207, 316]
        # In a real scenario, this helps in visual inspection of the plateau [cite: 281]
        rho_err[t] = math.sqrt(abs(2 * (1 + 2 * rho[1:t+1].sum()) / N_total))
        
    return rho, gamma, rho_err


def autocorrelation_proj_with_replicas(data_replicas, grad_f, max_lag):
    """
    Eq. (31): Implementation of the Gamma-method autocorrelation.
    """
    projected_reps = [rep @ grad_f for rep in data_replicas]
    
    # Calculate N_total and R
    N_total = sum(len(r) for r in projected_reps)
    R = len(projected_reps)
    
    all_projected = tr.cat(projected_reps)
    mean_global = all_projected.mean()
    
    # SAFETY GUARD: Ensure max_lag is smaller than the shortest replica
    # Wolff suggests max_lag should not exceed min(Nr)/2 [cite: 415]
    min_Nr = min(len(rep) for rep in projected_reps)
    actual_max_lag = min(max_lag, min_Nr - 1) 
    
    if actual_max_lag <= 0:
        raise ValueError(f"Replica length ({min_Nr}) is too short for the requested lag.")

    gamma = tr.zeros(actual_max_lag)
    
    for t in range(actual_max_lag):
        s_sum = 0.0
        for rep in projected_reps:
            if len(rep) > t:
                d1 = rep[:len(rep)-t] - mean_global
                d2 = rep[t:] - mean_global
                s_sum += (d1 * d2).sum()
        
        # Proper normalization Eq. (31) 
        denominator = N_total - R * t
        if denominator <= 0:
            # Fallback for edge cases where t is too large for the dataset
            gamma[t] = 0.0 
        else:
            gamma[t] = s_sum / denominator

    rho = gamma / gamma[0] # [cite: 433]
    
    # Statistical error calculation [cite: 434]
    rho_err = tr.zeros(actual_max_lag)
    for t in range(actual_max_lag):
        rho_err[t] = math.sqrt(abs(2 * (1 + 2 * rho[1:t+1].sum()) / N_total))
        
    return rho, gamma, rho_err


def find_optimal_window(rho, N, S, maxW):
    """
    Eq. (50)-(52): Automatic window selection procedure. [cite: 269, 274, 277]
    Balances systematic truncation error with statistical noise. [cite: 216, 290]
    """
    tau_hist, dtau_hist, W_hist = [], [], []
    
    for W in range(1, min(maxW, N // 2)):
        # Eq. (25): Integrated autocorrelation time at window W [cite: 128]
        tau_int = 0.5 + rho[1:W+1].sum().item()
        
        # Eq. (42): Statistical error of tau_int [cite: 212]
        var_tau = (4.0 / N) * (W + 0.5 - tau_int) * (tau_int**2)
        err_tau = math.sqrt(max(var_tau, 0))
        
        tau_hist.append(tau_int)
        dtau_hist.append(err_tau)
        W_hist.append(W)
        
        # Eq. (51): Estimate decay scale tau [cite: 271, 272]
        if tau_int <= 0.5:
            tau_hat = 1e-10
        else:
            tau_hat = (1.0 / tau_int + 1.0 / (12.0 * tau_int**3))**(-1)
        
        # Windowing criterion Eq. (52) [cite: 274, 277]
        # We use S * tau_hat as the hypothesis for the decay scale [cite: 280]
        if (math.exp(-W / (S * tau_hat)) - (S * tau_hat) / math.sqrt(W * N)) < 0:
            return W, tau_int, err_tau, tau_hist, dtau_hist, W_hist
            
    return W_hist[-1], tau_hist[-1], dtau_hist[-1], tau_hist, dtau_hist, W_hist

def gamma_method_with_replicas(data_replicas, f, S=1.5, max_lag=200):
    """
    Ulli Wolff's Gamma analysis for general scalar functionals. [cite: 15, 17]
    Handles multiple replicas, bias correction, and error of the error. [cite: 53, 110, 202]
    """
    all_data = tr.cat(data_replicas, dim=0)
    N_total = len(all_data)
    mean_a = all_data.mean(dim=0)
    
    # Gradient calculation using autograd
    grad_f = compute_gradient_autograd(f, mean_a)
    
    # Calculate ACF and variances using replica-aware logic
    rho, gamma, rho_err = autocorrelation_proj_with_replicas(data_replicas, grad_f, max_lag)
    
    # Automated selection of the summation window W Eq. (43) [cite: 217, 266]
    W_opt, tau_int, dtau_int, tau_hist, dtau_hist, W_hist = find_optimal_window(rho, N_total, S, max_lag)

    # Eq. (34): Naive variance disregarding autocorrelations [cite: 176, 179]
    v_F = gamma[0].item()
    
    # Eq. (49): Apply bias correction for the variance estimate [cite: 262, 265]
    bias_factor = (1.0 + (2.0 * W_opt + 1.0) / N_total)
    corrected_variance_F = (v_F * 2.0 * tau_int) * bias_factor
    
    # Eq. (23): Final statistical error of the derived quantity [cite: 119]
    dvalue = math.sqrt(corrected_variance_F / N_total)
    
    # Eq. (40): Error of the error estimate (ddvalue) [cite: 207, 218]
    ddvalue = dvalue * math.sqrt((W_opt + 0.5) / N_total)
    
    # Mean value of the derived quantity Eq. (14) [cite: 97]
    value = f(mean_a).item()

    # Eq. (27)-(29): Consistency check across replicas (Goodness of fit) [cite: 139, 142]
    if len(data_replicas) > 1:
        # Per-replica means and variance contribution
        F_r = tr.tensor([f(rep.mean(dim=0)).item() for rep in data_replicas])
        N_r = tr.tensor([len(rep) for rep in data_replicas])
        chi2 = tr.sum((F_r - value)**2 * N_r / (v_F * 2.0 * tau_int)).item()
        Q = 1 - chi2dist.cdf(chi2, df=len(data_replicas) - 1)
    else:
        Q = None

    return {
        "value": value,
        "dvalue": dvalue,
        "ddvalue": ddvalue,
        "tau_int": tau_int,
        "dtau_int": dtau_int,
        "W_opt": W_opt,
        "Q": Q,
        "acf": rho,
        "acf_error": rho_err,
        "tau_int_history": tau_hist,
        "dtau_int_history": dtau_hist,
        "W_hist": W_hist
    }

# --- Test Utilities (Simulating section C.2 of the paper) ---

def generate_autocorrelated_sequence(N, tau, seed=None):
    """
    Generates autocorrelated Gaussian sequence using AR(1) logic (Eq. 68). [cite: 445]
    """
    if seed is not None: tr.manual_seed(seed)
    eta = tr.randn(N)
    nu = tr.zeros(N)
    a = (2.0 * tau - 1.0) / (2.0 * tau + 1.0)
    nu[0] = eta[0]
    for i in range(1, N):
        nu[i] = math.sqrt(1 - a**2) * eta[i] + a * nu[i-1]
    return nu

def main():
    """
    Main execution simulating the log-mass test case (Section C.2). [cite: 438]
    """
    # Parameters from the paper test case [cite: 509, 510]
    m, tau1, tau2, q = 0.2, 4.0, 8.0, 0.2
    Nrep, Nperrep = 8, 1000
    G0, G1 = 1.0, math.exp(-m)
    
    data_replicas = []
    for r in range(Nrep):
        nu1 = generate_autocorrelated_sequence(Nperrep, tau1, seed=r)
        nu2 = generate_autocorrelated_sequence(Nperrep, tau2, seed=r+100)
        nu3 = generate_autocorrelated_sequence(Nperrep, tau2, seed=r+200)
        
        a1 = G0 + q * (nu1 + nu2)
        a2 = G1 + q * (nu1 + nu3)
        data_replicas.append(tr.stack([a1, a2], dim=1))

    # Observable: F = log(a1 / a2) [cite: 440]
    f = lambda A: tr.log(A[0] / A[1])

    results = gamma_method_with_replicas(data_replicas, f, S=1.5)

    print("--- Gamma-Method Analysis Results ---")
    print(f"F        = {results['value']:.6f} +/- {results['dvalue']:.6f} (stat error of error: +/- {results['ddvalue']:.6f})")
    print(f"tau_int  = {results['tau_int']:.3f} +/- {results['dtau_int']:.3f}")
    print(f"W_opt    = {results['W_opt']}")
    print(f"Q-value  = {results['Q']:.4f}" if results['Q'] is not None else "Q-value  = N/A (1 replicum)")

if __name__ == "__main__":
    main()