#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Created on Tue Jan 27 11:02:25 EST 2026

The model CP(2) sigma model in 2D. 

the action of the model is 
S = -\beta/3 \sim_x \sum_\mu |z(x)^\dagger*z(x+\mu)|
S = -\beta/3 \sim_x \sum_\mu Tr P(x) P(x+\mu) 
with P(x) = U(x) E_11 U(x)\dagger and U(x) in SU(3)
E_11 = diag(1,0,0)

we store  z \in C^3 whith z\dagger z = 1

The z field is the first column of the SU(3) matrix U

The model manifold is the coset [SU(3)/U(2)]^V where V the volume of the system

My field layout will be differend than O3.py, ie the last tensor index
is the spin index.

z is a tensor of shape [Nb, Lx, Ly, 3] with Nb being the batch size and Lx, Ly the lattice dimensions.

@author: Kostas Orginos
"""

import torch as tr
import numpy as np
from LieGroups import su3

class cp2():

    def _commutator_PM(self, z):
        # P(x) = z z^\dagger, M(x) = sum of neighbor projectors
        P = z.unsqueeze(-1) * tr.conj(z).unsqueeze(-2)
        M = (tr.roll(P, shifts=1, dims=1) + tr.roll(P, shifts=-1, dims=1) +
             tr.roll(P, shifts=1, dims=2) + tr.roll(P, shifts=-1, dims=2))
        return tr.matmul(P, M) - tr.matmul(M, P)

    def action(self,z):
        A = self.Nd * self.Vol * tr.ones(z.shape[0], device=self.device)
        # Normalize so aligned fields have zero action:
        # S = (beta/3) * [Nd*Vol - sum_{x,mu} |z(x)^\dagger z(x+mu)|^2]
        dims = tuple(range(1, 1 + self.Nd))
        for mu in range(self.Nd):
            z_shift = tr.roll(z, shifts=-1, dims=1 + mu)
            overlap = tr.einsum('...c,...c->...', tr.conj(z), z_shift)
            #overlap = tr.sum(tr.conj(z)*z_shift,dim=-1)
            A = A - tr.sum(tr.abs(overlap) ** 2, dim=dims)
        return (self.beta / 3.0) * A

    def action_unshifted(self, z):
        # S = -(beta/3) * sum_{x,mu} |z(x)^\dagger z(x+mu)|^2
        dims = tuple(range(1, 1 + self.Nd))
        A = tr.zeros(z.shape[0], device=self.device)
        for mu in range(self.Nd):
            z_shift = tr.roll(z, shifts=-1, dims=1 + mu)
            overlap = tr.einsum('...c,...c->...', tr.conj(z), z_shift)
            A = A - tr.sum(tr.abs(overlap) ** 2, dim=dims)
        return (self.beta / 3.0) * A

    # slower than the action
    def action_opt(self,z):
        # 2D-only fast path: keep complex multiply, avoid abs() sqrt.
        A = 2 * self.Vol * tr.ones(z.shape[0], device=self.device)
        z_x = tr.roll(z, shifts=-1, dims=1)
        z_y = tr.roll(z, shifts=-1, dims=2)
        ox = tr.sum(tr.conj(z) * z_x, dim=-1)
        oy = tr.sum(tr.conj(z) * z_y, dim=-1)
        A = A - tr.sum(
            ox.real * ox.real + ox.imag * ox.imag +
            oy.real * oy.real + oy.imag * oy.imag,
            dim=(1, 2),
        )
        return (self.beta / 3.0) * A

    
    def force_components(self, z):
        # F_a(x) = (beta/3) * tr(T_a [P(x), M(x)])
        comm = self._commutator_PM(z)
        Fa = (self.beta / 3.0) * tr.einsum('aij,...ji->...a', self.T, comm)
        return tr.real(Fa)

    def force_matrix(self, z):
        # F(x) = sum_a F_a(x) T_a  (anti-hermitian, traceless 3x3)
        Fa = self.force_components(z)
        return tr.einsum('...a,aij->...ij', Fa.to(self.T.dtype), self.T)

    def force(self, z):
        return self.force_components(z)

    #here I have a problem... how do I know the batch size?
    #I need to set it in the constructor...
    # again here I explicitely make it 2-d
    def refreshP_components(self):
        # real coefficients for the 8 generators
        P = tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1], 8],
                      dtype=self.dtype, device=self.device)
        return P

    def refreshP_matrix(self):
        # anti-hermitian traceless matrix momenta
        Pa = self.refreshP_components()
        return tr.einsum('...a,aij->...ij', Pa.to(self.T.dtype), self.T)

    def refreshP(self):
        return self.refreshP_components()

    def kinetic_components(self, P):
        # P: [B, Lx, Ly, 8] real coefficients
        return 0.5 * tr.sum(P * P, dim=(1, 2, 3))

    def kinetic_matrix(self, P):
        # P: [B, Lx, Ly, 3, 3] anti-hermitian, traceless
        # with su3.T normalization: tr(T_a T_b) = -1/2 delta_ab
        PP = tr.matmul(P, P)
        trPP = tr.diagonal(PP, dim1=-1, dim2=-2).sum(dim=-1)
        return -tr.sum(tr.real(trPP), dim=(1, 2))

    def kinetic(self, P):
        return self.kinetic_components(P)

    def evolveQ_matrix(self, dt, P, Q):
        # P is anti-hermitian matrix field [B, Lx, Ly, 3, 3]
        R = su3.expo(dt * P)
        return tr.einsum('...ij,...j->...i', R, Q)

    def evolveQ(self, dt, P, Q):
        # P is components [B, Lx, Ly, 8]
        Pm = tr.einsum('...a,aij->...ij', P.to(self.T.dtype), self.T)
        return self.evolveQ_matrix(dt, Pm, Q)
    
    def Q(self, z):
        # Geometric U(1) plaquette definition for CP(2) in 2D.
        if self.Nd != 2:
            print("Topological charge is defined in 2D only")
            return 0

        eps = tr.finfo(self.dtype).eps
        z_x = tr.roll(z, shifts=-1, dims=1)
        z_y = tr.roll(z, shifts=-1, dims=2)
        u_x = tr.einsum('...c,...c->...', tr.conj(z), z_x)
        u_y = tr.einsum('...c,...c->...', tr.conj(z), z_y)
        u_x = u_x / (tr.abs(u_x) + eps)
        u_y = u_y / (tr.abs(u_y) + eps)

        u_y_x = tr.roll(u_y, shifts=-1, dims=1)
        u_x_y = tr.roll(u_x, shifts=-1, dims=2)
        U_p = u_x * u_y_x * tr.conj(u_x_y) * tr.conj(u_y)
        theta = tr.angle(U_p)
        return tr.sum(theta, dim=(1, 2)) / (2.0 * np.pi)

    def Q_continuum(self, z):
        # Discretized continuum formula: Q = (1/2πi) sum Tr(P [∂x P, ∂y P])
        if self.Nd != 2:
            print("Topological charge is defined in 2D only")
            return 0
        P = z.unsqueeze(-1) * tr.conj(z).unsqueeze(-2)
        dPx = tr.roll(P, shifts=-1, dims=1) - P
        dPy = tr.roll(P, shifts=-1, dims=2) - P
        comm = tr.matmul(dPx, dPy) - tr.matmul(dPy, dPx)
        trPcomm = tr.diagonal(tr.matmul(P, comm), dim1=-1, dim2=-2).sum(dim=-1)
        q = tr.imag(trPcomm)
        return tr.sum(q, dim=(1, 2)) / (2.0 * np.pi)


    def coldStart(self):
        cdtype = tr.complex128 if self.dtype == tr.float64 else tr.complex64
        z = tr.zeros([self.Bs, self.V[0], self.V[1], self.N],
                     dtype=cdtype, device=self.device)
        z[..., 0] = 1.0 + 0.0j
        return z
        
    def hotStart(self):
        cdtype = tr.complex128 if self.dtype == tr.float64 else tr.complex64
        real = tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1], self.N],
                         dtype=self.dtype, device=self.device)
        imag = tr.normal(0.0, 1.0, [self.Bs, self.V[0], self.V[1], self.N],
                         dtype=self.dtype, device=self.device)
        z = tr.complex(real, imag).to(dtype=cdtype)
        # normalize each site to z^\dagger z = 1
        n = tr.sqrt(tr.sum(tr.abs(z) ** 2, dim=-1, keepdim=True))
        return z / n
    
    def __init__(self,V,beta,batch_size=1,device="cpu",dtype=tr.float32): 
            self.V = V # a tuple with the lattice dimensions
            self.Nd = len(V)
            self.Vol = np.prod(V)
            self.beta = beta # the coupling
            self.device=device
            self.dtype=dtype
            self.cdtype = tr.complex128 if self.dtype == tr.float64 else tr.complex64
            self.Bs=batch_size # batch size
            self.N = 3 # only the cp2 is simulated here
            # the generators of the group            
            self.T = su3.T.to(dtype=self.cdtype, device=self.device)

def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(description="cp2 tests")
    parser.add_argument("--test", action="append", default=[],
                        help="Test(s) to run: action, force, integrator, hmc, visualize, corr, all")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.263)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fd-eps", type=float, default=1e-6)
    parser.add_argument("--fd-sites", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--int-kmin", type=float, default=1.0)
    parser.add_argument("--int-kmax", type=float, default=3.0)
    parser.add_argument("--int-nk", type=int, default=50)
    parser.add_argument("--int-plot", action="store_true")
    parser.add_argument("--hmc-nmd", type=int, default=3)
    parser.add_argument("--hmc-nwarm", type=int, default=100)
    parser.add_argument("--hmc-nmeas", type=int, default=200)
    parser.add_argument("--hmc-nskip", type=int, default=10)
    parser.add_argument("--hmc-traj", type=float, default=1.0)
    parser.add_argument("--hmc-progress", action="store_true")
    parser.add_argument("--hmc-plot", action="store_true")
    parser.add_argument("--hmc-history", action="store_true")
    parser.add_argument("--corr-maxr", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if tr.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    dtype = tr.float64 if args.dtype == "float64" else tr.float32

    if not args.test:
        args.test = ["all"]
    tests = set()
    for t in args.test:
        if t == "all":
            tests.update(["action", "force", "integrator", "hmc", "visualize", "corr"])
        else:
            tests.add(t)

    def time_it(fn, z, Nwarm, Niter, Nrep):
        for _ in range(Nwarm):
            fn(z)
        times = []
        for _ in range(Nrep):
            tic = time.perf_counter()
            for _ in range(Niter):
                fn(z)
            toc = time.perf_counter()
            times.append((toc - tic) * 1.0e6 / Niter)
        times.sort()
        return times

    if "action" in tests:
        o = cp2([args.L, args.L], args.beta, batch_size=args.batch_size, device=device, dtype=dtype)
        z = o.hotStart()

        a0 = o.action(z)
        a1 = o.action_opt(z)
        diff = a0 - a1
        max_abs = tr.max(tr.abs(diff)).item()
        mean_abs = tr.mean(tr.abs(diff)).item()
        print(f"action vs action_opt allclose: {tr.allclose(a0, a1, rtol=1e-6, atol=1e-6)}")
        print(f"max|diff|={max_abs:.3e}, mean|diff|={mean_abs:.3e}")

        t0 = time_it(o.action, z, args.warmup, args.iters, args.repeats)
        t1 = time_it(o.action_opt, z, args.warmup, args.iters, args.repeats)
        print(f"action time [min/med/max] = {t0[0]:0.4f}/{t0[args.repeats//2]:0.4f}/{t0[-1]:0.4f} us")
        print(f"action_opt time [min/med/max] = {t1[0]:0.4f}/{t1[args.repeats//2]:0.4f}/{t1[-1]:0.4f} us")

    if "force" in tests:
        o2 = cp2([4, 4], args.beta, batch_size=1, device=device, dtype=tr.float64)
        z2 = o2.hotStart()
        Fm = o2.force_matrix(z2)
        herm_err = tr.norm(Fm + tr.conj(Fm).transpose(-1, -2)).item()
        tr_err = tr.norm(tr.diagonal(Fm, dim1=-1, dim2=-2).sum(dim=-1)).item()
        print(f"force_matrix anti-hermitian err={herm_err:.3e}, trace err={tr_err:.3e}")

        eps = args.fd_eps
        ncheck = args.fd_sites
        rng = tr.Generator(device=device)
        rng.manual_seed(args.seed)
        b_idx = tr.randint(0, o2.Bs, (ncheck,), generator=rng)
        x_idx = tr.randint(0, o2.V[0], (ncheck,), generator=rng)
        y_idx = tr.randint(0, o2.V[1], (ncheck,), generator=rng)

        Fa_all = o2.force_components(z2)
        errs = []
        for k in range(ncheck):
            b = int(b_idx[k])
            x0 = int(x_idx[k])
            y0 = int(y_idx[k])
            Fa = Fa_all[b, x0, y0]
            fd = tr.zeros_like(Fa)
            for a in range(8):
                Ta = o2.T[a]
                U = tr.linalg.matrix_exp(eps * Ta)
                Uinv = tr.linalg.matrix_exp(-eps * Ta)
                zp = z2.clone()
                zm = z2.clone()
                zp[b, x0, y0] = U @ zp[b, x0, y0]
                zm[b, x0, y0] = Uinv @ zm[b, x0, y0]
                Sp = o2.action(zp)[b]
                Sm = o2.action(zm)[b]
                fd[a] = (Sp - Sm) / (2.0 * eps)
            errs.append(tr.abs(fd + Fa))
        errs = tr.stack(errs, dim=0)
        fd_max = tr.max(errs).item()
        fd_mean = tr.mean(errs).item()
        print(f"finite-diff over {ncheck} sites: max|dS/dtheta + F_a|={fd_max:.3e}, mean={fd_mean:.3e}")

    if "integrator" in tests:
        import numpy as np
        import integrators as integ
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        oi = cp2([args.L, args.L], args.beta, batch_size=1, device=device, dtype=dtype)
        z = oi.hotStart()
        P = oi.refreshP()
        Hi = oi.kinetic(P) + oi.action(z)

        x = []
        y = []
        y2 = []
        for rk in np.logspace(args.int_kmin, args.int_kmax, args.int_nk):
            k = int(rk)
            dt = 1.0 / k
            l = integ.leapfrog(oi.force, oi.evolveQ, k, 1.0)
            l2 = integ.minnorm2(oi.force, oi.evolveQ, k, 1.0)
            PP, QQ = l.integrate(P, z)
            PP2, QQ2 = l2.integrate(P, z)
            Hf = oi.kinetic(PP) + oi.action(QQ)
            Hf2 = oi.kinetic(PP2) + oi.action(QQ2)
            DH = tr.abs(Hf - Hi).item()
            DH2 = tr.abs(Hf2 - Hi).item()
            x.append(dt * dt)
            y.append(DH)
            y2.append(DH2)
            print(f"dt={dt:.6f}  dH={DH:.3e}  dH2={DH2:.3e}")

        if args.int_plot and plt is not None:
            plt.plot(x, y, x, y2)
            plt.xlabel(r"$\epsilon^2$")
            plt.ylabel(r"$\Delta H$")
            plt.show()

    if "hmc" in tests:
        import numpy as np
        import integrators as integ
        import update as u
        from analysis.autocorr import summary as ac_summary
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        oh = cp2([args.L, args.L], args.beta, batch_size=args.batch_size, device=device, dtype=dtype)
        z = oh.hotStart()
        I = integ.minnorm2(oh.force, oh.evolveQ, args.hmc_nmd, args.hmc_traj)
        hmc = u.hmc(T=oh, I=I, verbose=False)

        warm_it = range(args.hmc_nwarm)
        if args.hmc_progress and tqdm is not None:
            warm_it = tqdm(warm_it, desc="warmup")
        for _ in warm_it:
            z = hmc.evolve(z, 1)
        print(f"HMC warmup acceptance: {hmc.calc_Acceptance():.3f}")
        hmc.AcceptReject = []

        E = []
        E0 = []
        Q = []
        Qc = []
        Q_bug = []
        it = range(args.hmc_nmeas)
        if args.hmc_progress and tqdm is not None:
            it = tqdm(it)
        for _ in it:
            E.extend((oh.action(z) / oh.Vol).tolist())
            E0.extend((oh.action_unshifted(z) / oh.Vol).tolist())
            Q.extend(oh.Q(z).tolist())
            Qc.extend(oh.Q_continuum(z).tolist())
            # sanity: |U_p| should be ~1
            u_x = tr.einsum('...c,...c->...', tr.conj(z), tr.roll(z, shifts=-1, dims=1))
            u_y = tr.einsum('...c,...c->...', tr.conj(z), tr.roll(z, shifts=-1, dims=2))
            u_x = u_x / (tr.abs(u_x) + tr.finfo(oh.dtype).eps)
            u_y = u_y / (tr.abs(u_y) + tr.finfo(oh.dtype).eps)
            u_y_x = tr.roll(u_y, shifts=-1, dims=1)
            u_x_y = tr.roll(u_x, shifts=-1, dims=2)
            U_p = u_x * u_y_x * tr.conj(u_x_y) * tr.conj(u_y)
            Q_bug.append(tr.max(tr.abs(tr.abs(U_p) - 1.0)).item())
            z = hmc.evolve(z, args.hmc_nskip)
            if args.hmc_progress and tqdm is not None:
                it.set_postfix({"E": f"{np.mean(E):.3f}", "Q": f"{np.mean(Q):.3f}"})

        print(f"HMC acceptance: {hmc.calc_Acceptance():.3f}")
        if len(E) > 0:
            print(f"<E> = {np.mean(E):.6f} +/- {np.std(E, ddof=1)/np.sqrt(len(E)):.6f}")
            print(f"<E_unshifted> = {np.mean(E0):.6f} +/- {np.std(E0, ddof=1)/np.sqrt(len(E0)):.6f}")
        if len(Q) > 0:
            print(f"<Q> = {np.mean(Q):.6f} +/- {np.std(Q, ddof=1)/np.sqrt(len(Q)):.6f}")
            print(f"<Q_continuum> = {np.mean(Qc):.6f} +/- {np.std(Qc, ddof=1)/np.sqrt(len(Qc)):.6f}")
        if len(Q_bug) > 0:
            print(f"max| |U_p|-1 | = {np.max(Q_bug):.3e}")

        def autocorr(x):
            x = np.asarray(x, dtype=float)
            x = x - np.mean(x)
            n = len(x)
            if n < 2:
                return np.array([1.0])
            corr = np.correlate(x, x, mode="full")[n-1:]
            return corr / corr[0]

        def tau_int(x, maxlag=100):
            c = autocorr(x)
            maxlag = min(maxlag, len(c) - 1)
            return 0.5 + np.sum(c[1:maxlag+1])

        if len(E) > 2:
            sE = ac_summary(E)
            print(f"tau_int(E) ~ {sE['tau_int']:.2f}, ESS ~ {sE['ess']:.1f}")
        if len(Q) > 2:
            sQ = ac_summary(Q)
            print(f"tau_int(Q) ~ {sQ['tau_int']:.2f}, ESS ~ {sQ['ess']:.1f}")
        if len(Qc) > 2:
            sQc = ac_summary(Qc)
            print(f"tau_int(Q_continuum) ~ {sQc['tau_int']:.2f}, ESS ~ {sQc['ess']:.1f}")

        if args.hmc_plot and plt is not None:
            plt.plot(range(len(E)), E)
            plt.xlabel("traj")
            plt.ylabel("E")
            plt.show()
            plt.plot(range(len(Q)), Q)
            plt.xlabel("traj")
            plt.ylabel("Q")
            plt.show()
        if args.hmc_history and plt is not None:
            plt.plot(range(len(E)), E)
            plt.xlabel("traj")
            plt.ylabel("E")
            plt.show()
            plt.plot(range(len(Q)), Q)
            plt.xlabel("traj")
            plt.ylabel("Q")
            plt.show()

    if "visualize" in tests:
        import numpy as np
        import integrators as integ
        import update as u
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import hsv_to_rgb
        except Exception:
            plt = None
            hsv_to_rgb = None
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        if plt is None or hsv_to_rgb is None:
            print("matplotlib is required for visualization")
            return

        ov = cp2([args.L, args.L], args.beta, batch_size=1, device=device, dtype=dtype)
        z = ov.hotStart()
        I = integ.minnorm2(ov.force, ov.evolveQ, args.hmc_nmd, args.hmc_traj)
        hmc = u.hmc(T=ov, I=I, verbose=False)
        warm_it = range(args.hmc_nwarm)
        if args.hmc_progress and tqdm is not None:
            warm_it = tqdm(warm_it, desc="warmup")
        for _ in warm_it:
            z = hmc.evolve(z, 1)
        print(f"Visualize warmup acceptance: {hmc.calc_Acceptance():.3f}")
        Qb = ov.Q(z).cpu().numpy()
        print("Visualize Q after warmup (per batch):")
        print(Qb)

        z0 = z[0]
        # 1) RGB from projector diagonal (|z_i|^2)
        amps = tr.real(tr.abs(z0) ** 2).cpu().numpy()  # [Lx, Ly, 3]
        rgb_diag = amps / (amps.max() + 1e-12)

        # 2) RGB from amps with brightness from arg(P01)
        p01 = z0[..., 0] * tr.conj(z0[..., 1])
        phase01 = tr.angle(p01).cpu().numpy()
        brightness = 0.5 + 0.5 * np.cos(phase01)
        rgb_phase_bright = rgb_diag * brightness[..., None]

        # 3) Hue from arg(P01), value from |P01|
        mag01 = tr.abs(p01).cpu().numpy()
        hue = (phase01 + np.pi) / (2.0 * np.pi)
        hsv = np.stack([hue, np.ones_like(hue), mag01 / (mag01.max() + 1e-12)], axis=-1)
        rgb_hsv = hsv_to_rgb(hsv)

        # 4) Plaquette angle map from induced U(1) links
        u_x = tr.einsum('...c,...c->...', tr.conj(z0), tr.roll(z0, shifts=-1, dims=0))
        u_y = tr.einsum('...c,...c->...', tr.conj(z0), tr.roll(z0, shifts=-1, dims=1))
        u_x = u_x / (tr.abs(u_x) + tr.finfo(ov.dtype).eps)
        u_y = u_y / (tr.abs(u_y) + tr.finfo(ov.dtype).eps)
        u_y_x = tr.roll(u_y, shifts=-1, dims=0)
        u_x_y = tr.roll(u_x, shifts=-1, dims=1)
        U_p = u_x * u_y_x * tr.conj(u_x_y) * tr.conj(u_y)
        theta = tr.angle(U_p).cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0, 0].imshow(rgb_diag)
        axes[0, 0].set_title("diag(P)=|z|^2 (RGB)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(rgb_phase_bright)
        axes[0, 1].set_title("RGB amps + phase(P01) brightness")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(rgb_hsv)
        axes[1, 0].set_title("arg(P01) hue, |P01| value")
        axes[1, 0].axis("off")

        im = axes[1, 1].imshow(theta, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, 1].set_title("plaquette angle")
        axes[1, 1].axis("off")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    if "corr" in tests:
        import numpy as np
        import integrators as integ
        import update as u
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        oc = cp2([args.L, args.L], args.beta, batch_size=args.batch_size, device=device, dtype=dtype)
        z = oc.hotStart()
        I = integ.minnorm2(oc.force, oc.evolveQ, args.hmc_nmd, args.hmc_traj)
        hmc = u.hmc(T=oc, I=I, verbose=False)

        warm_it = range(args.hmc_nwarm)
        if args.hmc_progress and tqdm is not None:
            warm_it = tqdm(warm_it, desc="warmup")
        for _ in warm_it:
            z = hmc.evolve(z, 1)
        print(f"Corr warmup acceptance: {hmc.calc_Acceptance():.3f}")
        hmc.AcceptReject = []

        maxr = args.corr_maxr if args.corr_maxr is not None else args.L // 2
        Csum = np.zeros(maxr, dtype=float)
        nconf = 0
        Qs = []
        xi2_list = []

        def projector_traceless(zcfg):
            P = zcfg.unsqueeze(-1) * tr.conj(zcfg).unsqueeze(-2)
            eye = tr.eye(3, dtype=P.dtype, device=P.device)
            return P - eye / 3.0

        it = range(args.hmc_nmeas)
        if args.hmc_progress and tqdm is not None:
            it = tqdm(it, desc="meas")
        for _ in it:
            X = projector_traceless(z)  # [B,L,L,3,3]
            # chi via zero-momentum mode
            Xavg = tr.mean(X, dim=(1, 2))
            chi = oc.Vol * tr.real(tr.einsum('...ij,...ji->...', Xavg, tr.conj(Xavg)))

            # C2p via p=(2π/L,0)
            phase = tr.exp(1j * tr.arange(oc.V[0], device=X.device) * (2.0 * np.pi / oc.V[0]))
            phase = phase.view(1, oc.V[0], 1, 1, 1)
            Xp = tr.mean(X * phase, dim=(1, 2))
            C2p = oc.Vol * tr.real(tr.einsum('...ij,...ji->...', Xp, tr.conj(Xp)))

            xi2 = (1.0 / (4.0 * np.sin(np.pi / oc.V[0]) ** 2)) * (chi / C2p - 1.0)
            xi2_list.extend(xi2.cpu().numpy().tolist())

            # two-point function along x-direction
            X0 = X
            for r in range(maxr):
                Xr = tr.roll(X0, shifts=-r, dims=1)
                trXX = tr.real(tr.einsum('...ij,...ji->...', X0, Xr))
                Csum[r] += tr.mean(trXX).item()

            Qs.extend(oc.Q(z).tolist())
            nconf += 1
            z = hmc.evolve(z, args.hmc_nskip)

        C = Csum / max(1, nconf)
        chi_t = np.mean(np.array(Qs) ** 2) / oc.Vol
        print(f"chi_t = {chi_t:.6e}")
        if len(xi2_list) > 0:
            print(f"<xi^2> = {np.mean(xi2_list):.6f}")
        print("C(r) (x-direction avg):")
        print(C)

    
if __name__ == "__main__":
   main()
    
            
