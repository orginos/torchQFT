
import math
import torch as tr
import numpy as np
import time

"""
Created on Wed Nov 2 03:30:00 2025

@author: Yamil Cahuana

"""

def get_observables_hist(sg,hmc, phi, Nwarm, Nmeas, Nskip,pp="no"):

    tic=time.perf_counter()
    Vol=sg.Vol
    lat=[phi.shape[1], phi.shape[2]]
    toc=time.perf_counter()

    print(f"time {(toc - tic)*1.0e6/Nwarm:0.4f} micro-seconds per HMC trajecrory")

    lC2p = []
    lchi_m = []
    E = []
    av_phi = []
    phase=tr.tensor(np.exp(1j*np.indices(tuple(lat))[0]*2*np.pi/lat[0]),dtype=sg.dtype,device=sg.device)
    for k in range(Nmeas):
        ttE = sg.action(phi)/Vol
        E.append(ttE)
        av_sigma = tr.mean(phi.view(sg.Bs,Vol),axis=1)
        av_phi.append(av_sigma)
        chi_m = av_sigma*av_sigma*Vol
        p1_av_sig = tr.mean(phi.view(sg.Bs,Vol)*phase.view(1,Vol),axis=1)
        C2p = tr.real(tr.conj(p1_av_sig)*p1_av_sig)*Vol
        if(k%100==0) and pp=="print":
            print("k= ",k,"(av_phi,chi_m, c2p, E) ", av_sigma.mean().numpy(),chi_m.mean().numpy(),C2p.mean().numpy(),ttE.mean().numpy())
            print("len(C2p): ", len(C2p))
        lC2p.append(C2p)
        lchi_m.append(chi_m)
        ## HMC update but also V cycle
        phi = hmc.evolve(phi,Nskip)

    return lC2p, lchi_m, E, av_phi, phi

### Gamma method implementation

def compute_gradient_autograd(f, mean_a):
    """
    Eq. (17): Compute gradient f_alpha = ∂f/∂A_alpha using autograd at the mean point.
    """
    mean_a = mean_a.clone().detach().requires_grad_(True)
    f_value = f(mean_a)
    f_value.backward()
    return mean_a.grad.detach()

def project_observable(data: tr.Tensor, grad_f: tr.Tensor) -> tr.Tensor:
    """
    Eq. (37): Project a_i^alpha to a_i^f = sum_alpha f_alpha * a_i^alpha
    """
    return data @ grad_f

def autocorrelation_proj_with_error(data: tr.Tensor, max_lag: int) -> tuple:
    """
    Eq. (31), (33) and the error estimation comes from Lüscher:
    Estimate normalized autocorrelation function rho(t) = Gamma_F(t)/Gamma_F(0)
    and its statistical error.
    """
    N = len(data)
    mean = data.mean()
    var = data.var(unbiased=True)
    acf = tr.zeros(max_lag)
    err = tr.zeros(max_lag)

    for t in range(max_lag):
        d1 = data[:N - t] - mean
        d2 = data[t:] - mean
        acf[t] = (d1 * d2).mean()

    acf /= var

    for t in range(max_lag):
        temp = 0.0
        for k in range(1, max_lag - t):
            term = acf[k + t] + acf[k - t] - 2 * acf[k] * acf[t]
            temp += term * term
        err[t] = math.sqrt(temp / N)

    return acf, err


def integrated_autocorrelation_time(acf: tr.Tensor, W: int) -> float:
    """
    Eq. (25): Truncated estimate of tau_int,F
    """
    return 0.5 + acf[1:W+1].sum().item()

def compute_tau_hat(tau_int: float) -> float:
    """
    Eq. (51): Estimate decay scale tau from tau_int using logarithmic expansion.
    """
    if tau_int <= 0.5:
        return 1e-10
    return (1.0 / tau_int + 1.0 / (12.0 * tau_int**3))**(-1) 

def g_function(W: int, tau_hat: float, N: int) -> float:
    """
    Eq. (52): Window optimization criterion.
    """
    return math.exp(-W / tau_hat) - tau_hat / math.sqrt(W * N)

def find_optimal_window(acf: tr.Tensor, N: int, S: float = 1.5, maxW: int = 100):
    """
    Eq. (43)–(52): Automatic window selection.
    Returns W_opt, tau_int[W], error[tau_int[W]], and full history of tau_int(W) and its error.
    """
    tau_values = []
    tau_errors = []
    W_hist = []
    for W in range(1, min(maxW, N // 2)):
        tau_int = integrated_autocorrelation_time(acf, W)
        var_tau = 4 / N * (W + 0.5 - tau_int) * tau_int**2
        err_tau = math.sqrt(max(var_tau, 0))
        tau_values.append(tau_int)
        tau_errors.append(err_tau)
        W_hist.append(W)
        tau_hat = compute_tau_hat(tau_int)*S
        if g_function(W, tau_hat, N) < 0:
            print(f"Optimal window found: W = {W}")
            return W, tau_int, err_tau, tau_values, tau_errors, W_hist
    last_W = min(maxW, N // 2)
    tau_int = integrated_autocorrelation_time(acf, last_W)
    var_tau = 4 / N * (last_W + 0.5 - tau_int) * tau_int**2
    err_tau = math.sqrt(max(var_tau, 0))
    tau_values.append(tau_int)
    tau_errors.append(err_tau)
    W_hist.append(last_W)
    return last_W, tau_int, err_tau, tau_values, tau_errors, W_hist

def split_first_dim_to_list(tensor: tr.Tensor) -> list:
    """
    Given a tensor of shape [N, T, D], returns a list of N tensors of shape [T, D].

    Args:
        tensor (torch.Tensor): A 3D tensor with shape [N, T, D].

    Returns:
        list of torch.Tensor: List of N tensors each with shape [T, D].
    """
    return [tensor[i] for i in range(tensor.shape[0])]

def gamma_method_with_replicas(data_replicas: list, f, S: float = 1.0, max_lag: int = 200):
    """
    Uli Wolff's Gamma analysis for arbitrary general scalar functional F(A, B, C, ...)
    if we want to do a analysis for a principal observable A, we can just set F(A, B, C, ...) = A.
    and autograd will take care of the gradient which is 1
    Returns:
        - acf: normalized autocorrelation function
        - acf_error: statistical error of acf
        - tau_int: integrated autocorrelation time
        - dtau_int: error of tau_int
        - tau_int_history: list of tau_int(W)
        - dtau_int_history: corresponding error bars
        - W_hist: list of W values for each tau_int(W)
        - W_opt: optimal window
        - value: estimate of F = f(A)
        - dvalue: error of value
        - ddvalue: error of the error estimate
        - Q: goodness-of-fit across replicas (Eq. 27–30) I still have to fix it 
    """
    all_data = tr.cat(data_replicas, dim=0)
    mean_a = all_data.mean(dim=0)
    grad_f = compute_gradient_autograd(f, mean_a)
    projected = tr.cat([project_observable(rep, grad_f) for rep in data_replicas])

    #max_lag = 200 # this is the maximum lag we will consider, IT IS
    acf_avg, acf_err = autocorrelation_proj_with_error(projected, max_lag=max_lag)
    W_opt, tau_int, dtau_int, tau_hist, dtau_hist, W_hist = find_optimal_window(acf_avg, len(projected), S=S, maxW=max_lag)

    var_f = acf_avg[0] * (2 * tau_int / len(projected))
    dvalue = math.sqrt(var_f.item())
    ddvalue = 0.5 * dvalue * dtau_int / tau_int if tau_int > 0 else 0.0
    value = f(mean_a).item()

    F_r = tr.tensor([f(rep.mean(dim=0)).item() for rep in data_replicas])
    N_r = tr.tensor([len(rep) for rep in data_replicas])
    chi2 = tr.sum((F_r - value)**2 * N_r / var_f).item()
    from scipy.stats import chi2 as chi2dist
    Q = 1 - chi2dist.cdf(chi2, df=len(data_replicas) - 1) if len(data_replicas) > 1 else None

    return {
        "acf": acf_avg,
        "acf_error": acf_err,
        "tau_int": tau_int,
        "dtau_int": dtau_int,
        "tau_int_history": tau_hist,
        "dtau_int_history": dtau_hist,
        "W_hist": W_hist,
        "W_opt": W_opt,
        "value": value,
        "dvalue": dvalue,
        "ddvalue": ddvalue,
        "Q": Q
    }


def generate_autocorrelated_sequence(N, tau, seed=None):
    """
    Generates an autocorrelated Gaussian sequence using an AR(1)-like process:
    
        nu[0] = eta[0]
        nu[i+1] = sqrt(1 - a^2) * eta[i+1] + a * nu[i]

    where:
        a = (2*tau - 1)/(2*tau + 1)
    
    Parameters:
        N (int): number of samples
        tau (float): target autocorrelation time
        seed (int or None): for reproducibility

    Returns:
        Tensor of shape [N]
    """
    if seed is not None:
        tr.manual_seed(seed)

    eta = tr.randn(N)
    nu = tr.zeros(N)
    a = (2 * tau - 1) / (2 * tau + 1)
    nu[0] = eta[0]
    for i in range(1, N):
        nu[i] = math.sqrt(1 - a ** 2) * eta[i] + a * nu[i - 1]
    return nu

def simulate_logmass_testcase(Nrep=8, Nperrep=1000, m=0.2, tau1=4, tau2=8, q=0.2, seed=123):
    """
    Simulates the example in Wolff's paper (section: Test in a simulator).
    Generates 8 replicas of 1000 measurements each for the observable:

        F = log(G(0)/G(1))  with  G(z) = exp(-mz)

    The two primary observables (a1, a2) are:
        a1_i = G(0) + q * (nu1_i + nu2_i)
        a2_i = G(1) + q * (nu1_i + nu3_i)

    Where each nu_j is an autocorrelated Gaussian noise sequence with specified tau.
    
    Returns:
        data_replicas: list of 8 tensors, each of shape [1000, 8]
    """
    G0 = math.exp(-m * 0)
    G1 = math.exp(-m * 1)
    data_replicas = []

    for r in range(Nrep):
        seed_offset = seed + r * 100
        nu1 = generate_autocorrelated_sequence(Nperrep, tau1, seed=seed_offset)
        nu2 = generate_autocorrelated_sequence(Nperrep, tau2, seed=seed_offset + 1)
        nu3 = generate_autocorrelated_sequence(Nperrep, tau2, seed=seed_offset + 2)

        a1 = G0 + q * (nu1 + nu2)
        a2 = G1 + q * (nu1 + nu3)

        replica = tr.stack([a1, a2], dim=1)  # shape [Nperrep, 2]
        data_replicas.append(replica)

    return data_replicas

def print_results(results):
    print("Gamma-method results:")
    print(f"F = {results['value']:.6f} ± {results['dvalue']:.6f} (±{results['ddvalue']:.6f})")
    print(f"tau_int = {results['tau_int']:.3f} ± {results['dtau_int']:.3f}")
    print(f"W_opt = {results['W_opt']}, Q = {results['Q']}")

import matplotlib.pyplot as plt

def plot_figure_2(results, max_lag=40):
    """
    Reproduce Fig. 2 from Wolff (2006) using output of gamma_method_with_replicas.
    Includes:
    - Top: autocorrelation function rho(t) with error bars.
    - Bottom: tau_int(W) with error bars.
    """
    acf = results["acf"]
    acf_error = results["acf_error"]
    tau_all = results["tau_int_history"]
    dtau_all = results["dtau_int_history"]
    tau= results["tau_int"]
    dtau= results["dtau_int"]
    W_vals = results["W_hist"]
    W_opt = results["W_opt"]
    tau_opt = results["tau_int"]

    fig, axs = plt.subplots(2, 1, figsize=(16, 16), sharex=False)

    # Top plot: autocorrelation rho(t)
    axs[0].errorbar(
        range(min(len(acf), max_lag)),
        acf[:max_lag],
        yerr=acf_error[:max_lag],
        fmt='o', capsize=3, label=r"$\rho(t)$"
    )
    axs[0].set_title("Normalized autocorrelation of observable")
    axs[0].set_xlabel(r"$\tau$")
    axs[0].set_ylabel(r"$\rho(t)$")
    axs[0].grid(True)

    # Bottom plot: tau_int(W)
    axs[1].errorbar(
        W_vals,
        tau_all,
        yerr=dtau_all,
        fmt='s', capsize=3,
        label=r"$\tau_{\mathrm{int}}(W)$"
    )
    axs[1].axvline(W_opt, color='r', linestyle='--', label=f"Optimal $W = {W_opt}$")
    axs[1].axhline(tau_opt, color='g', linestyle=':', label=fr"$\tau_{{\mathrm{{int}}}}$ = {tau_opt:.2f} $\pm${dtau:.2f}")
    axs[1].set_xlabel("$W$")
    axs[1].set_ylabel(r"$\tau_{\mathrm{int}}(W)$")
    axs[1].set_title("Integrated autocorrelation time")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    data_replicas = simulate_logmass_testcase()
    print(tr.stack(data_replicas).shape)

    # Define observable: F = log(a1 / a2)
    f = lambda A: tr.log(A[0] / A[1])

    # run Γ-method analysis
    results = gamma_method_with_replicas(data_replicas, f)

    # print results
    print(f"F = {results['value']:.6f} ± {results['dvalue']:.6f} (±{results['ddvalue']:.6f})")
    print(f"tau_int = {results['tau_int']:.3f} ± {results['dtau_int']:.3f}")
    print(f"W_opt = {results['W_opt']}, Q = {results['Q']}")

    # plot figure 2
    plot_figure_2(results)
    

if __name__ == "__main__":
    main()