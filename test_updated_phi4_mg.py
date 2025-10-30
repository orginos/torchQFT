#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test driver for the multigrid normalizing flow stack.
Correct Jacobian test with a short pretraining warmup so the Jacobian is non-trivial.
"""

from __future__ import annotations
import argparse, time, sys
import torch as tr
import torch.nn.functional as F

from updated_phi4_mg import MGflow, RGlayer, FlowBijector
from updated_stacked_model import StackedModel

def pick_device(name: str):
    name = name.lower()
    if name == "cuda" and tr.cuda.is_available():
        return tr.device("cuda")
    if name == "mps":
        try:
            if getattr(tr.backends, "mps", None) and tr.backends.mps.is_available():
                return tr.device("mps")
        except Exception:
            pass
    if name == "cpu":
        return tr.device("cpu")
    if tr.cuda.is_available():
        return tr.device("cuda")
    return tr.device("cpu")

def tnow(): return time.perf_counter()

def assert_close(a: tr.Tensor, b: tr.Tensor, atol=1e-5, rtol=1e-5, name="tensor"):
    max_abs = (a - b).abs().max().item()
    rel = max_abs / (b.abs().max().item() + 1e-12)
    ok = (max_abs <= atol + rtol * b.abs().max().item())
    return ok, max_abs, rel

def test_rg_layer(L=16, device=tr.device('cpu')):
    print("\n[TEST] RGlayer coarsen/refine")
    errs = {}
    for typ in ["average", "select"]:
        rg = RGlayer(typ)
        x = tr.randn(8, L, L, device=device)
        c, r = rg.coarsen(x)
        xr = rg.refine(c, r)
        ok, emax, rel = assert_close(xr, x, atol=1e-6, rtol=1e-6, name=f"rg({typ})")
        errs[typ] = (ok, emax, rel)
        print(f"  type={typ}: ok={ok} max_abs={emax:.3e} rel={rel:.3e}")
    return all(v[0] for v in errs.values())

def make_mg(L, width, layers, steps, device):
    bij_factory = FlowBijector(n_layers=layers, width=width)
    rg = RGlayer("average")
    return MGflow(size=(L, L), bijector_factory=bij_factory, rg=rg, Nconvs=steps).to(device)

@tr.no_grad()
def test_mgflow_roundtrip(L=16, batch=8, width=128, layers=2, steps=1, runs=10, device=tr.device('cpu')):
    print("\n[TEST] MGflow round-trip and log-prob identity")
    mg = make_mg(L, width, layers, steps, device)
    t_g = t_f = t_lp = 0.0
    passed = True
    for _ in range(runs):
        z = mg.prior_sample(batch, device=device)
        t0 = tnow(); x = mg.g(z); t_g += (tnow() - t0)
        t0 = tnow(); z2, ldj = mg.f(x); t_f += (tnow() - t0)
        ok1, emax1, _ = assert_close(z2, z, atol=1e-5, rtol=1e-5, name="z2~z")
        passed = passed and ok1
        t0 = tnow(); lp_x = mg.log_prob(x); t_lp += (tnow() - t0)
        lp_ref = mg.prior_log_prob(z) + ldj
        ok2, emax2, _ = assert_close(lp_x, lp_ref, atol=1e-4, rtol=1e-4, name="logprob identity")
        passed = passed and ok2
    print(f"  Round-trip z->x->z error ok={passed}")
    print(f"  Avg timings over {runs} runs (batch={batch}, L={L}):")
    print(f"    g(z):       {t_g/runs:.6f} s")
    print(f"    f(x):       {t_f/runs:.6f} s")
    print(f"    log_prob(x):{t_lp/runs:.6f} s")
    return passed

def warmup_flow(mg: MGflow, L: int, device: tr.device, steps: int = 10, batch: int = 8, lr: float = 1e-3):
    """Tiny supervised warmup so the flow deviates from identity (no phi4 needed)."""
    mg.train()
    opt = tr.optim.Adam(mg.parameters(), lr=lr, betas=(0.9, 0.99))
    target = tr.randn(1, L, L, device=device, dtype=next(mg.parameters()).dtype)
    for _ in range(steps):
        z = mg.prior_sample(batch, device=device).to(dtype=target.dtype)
        x = mg.g(z)
        loss = F.mse_loss(x, target.expand_as(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    mg.eval()
    return mg

def test_jacobian_autodiff(
    L=4, width=64, layers=1, steps=1, device=tr.device('cpu'),
    pretrain_steps: int = 10, pretrain_batch: int = 8, pretrain_lr: float = 1e-3
):
    print("\n[TEST] Autodiff Jacobian (tiny lattice) with short warmup")
    mg = make_mg(L, width, layers, steps, device).double()
    if pretrain_steps > 0:
        warmup_flow(mg, L=L, device=device, steps=pretrain_steps, batch=pretrain_batch, lr=pretrain_lr)

    x = tr.randn(1, L, L, device=device, dtype=tr.float64, requires_grad=True)

    def f_flat(v_flat):
        z, _ = mg.f(v_flat.view(1, L, L))
        return z.view(-1)  # length n = L*L

    t0 = tnow()
    J = tr.autograd.functional.jacobian(f_flat, x, vectorize=True)
    dt = tnow() - t0
    n = L * L
    J2 = J.reshape(n, n).detach()
    s, logabsdet = tr.slogdet(J2)

    # Code ldj (any shape) -> scalar
    _, ldj = mg.f(x)
    ldj_scalar = ldj.reshape(-1).sum()

    ok = tr.allclose(ldj_scalar.to(logabsdet.dtype), logabsdet, atol=1e-4, rtol=1e-4)
    diff = (ldj_scalar - logabsdet).abs().item()
    print(f"  ldj shape={tuple(ldj.shape)}, slogdet(J_f) = {logabsdet.item():.6f}, code ldj(sum) = {ldj_scalar.item():.6f}, |diff|={diff:.3e}, ok={bool(ok)} (time={dt:.3f}s)")
    return bool(ok)

@tr.no_grad()
def test_sampling(L=16, batch=8, device=tr.device('cpu')):
    print("\n[TEST] Sampling")
    mg = make_mg(L=L, width=64, layers=2, steps=1, device=device)
    x = mg.sample(batch)
    ok = (x.shape == (batch, L, L)) and (x.device.type == device.type)
    print(f"  sample shape {tuple(x.shape)}, device={x.device}, ok={ok}")
    return ok

def safe_import_phi4():
    try:
        import phi4 as p
        return p
    except Exception:
        print("  (phi4 not found; using zero action)")
        class P0:
            def phi4(self, shape, lam, mass, batch_size):
                class Obj:
                    def action(self, x): return tr.zeros(x.shape[0], device=x.device)
                return Obj()
        return P0()

def tiny_train_step(device, L=16, stages=2, batch=4, superbatch=2, width=64, layers=2):
    print("\n[TEST] StackedModel tiny train step (gradient accumulation)")
    p = safe_import_phi4()
    lam, mass = 0.5, -0.2
    o = p.phi4([L, L], lam, mass, batch_size=batch)

    def make_stage():
        return make_mg(L=L, width=width, layers=layers, steps=1, device=device)

    stages_list = [make_stage() for _ in range(stages)]
    model = StackedModel(stages_list, action_fn=lambda x: o.action(x)).to(device)

    opt = tr.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    accum = superbatch
    micro_bs = batch
    use_amp = (device.type == 'cuda')
    scaler = tr.amp.GradScaler('cuda', enabled=use_amp)
    loss_sum = 0.0
    last_z = None

    for _ in range(accum):
        z = model.prior_sample(micro_bs, device=device); last_z = z
        x = model.g(z)
        loss = (model.log_prob(x) + model.action_fn(x)).mean() / accum
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            tr.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            tr.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        loss_sum += float(loss.detach())

    print(f"  tiny-train accumulated loss: {loss_sum*accum:.6f}")
    # Save/load roundtrip
    save_path = "tmp_stacked_model.pth"
    tr.save({'state_dict': model.state_dict(),
             'meta': {'L': L, 'stages': stages, 'width': width, 'layers': layers}}, save_path)
    with tr.no_grad():
        x_ref = model.g(last_z)
    stages2 = [make_stage() for _ in range(stages)]
    model2 = StackedModel(stages2, action_fn=lambda x: o.action(x)).to(device)
    sd = tr.load(save_path, map_location=device)['state_dict']
    model2.load_state_dict(sd, strict=True)
    with tr.no_grad():
        x2 = model2.g(last_z)
        ok, emax, _ = assert_close(x2, x_ref, atol=1e-6, rtol=1e-6, name="reload forward match")
    print(f"  save/load forward match ok={ok}, max_abs={emax:.3e}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--pretrain-steps", type=int, default=10, help="warmup steps before Jacobian test")
    ap.add_argument("--pretrain-batch", type=int, default=8, help="warmup batch size")
    ap.add_argument("--pretrain-lr", type=float, default=1e-3, help="warmup LR")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    ok_all = True
    t0 = tnow(); ok = test_rg_layer(L=args.L, device=device); t1 = tnow()
    ok_all &= ok; print(f"  Time: {t1-t0:.6f}s")

    t0 = tnow(); ok = test_mgflow_roundtrip(L=args.L, batch=args.batch, width=args.width,
                                            layers=args.layers, steps=args.steps, runs=args.runs, device=device); t1 = tnow()
    ok_all &= ok; print(f"  Time: {t1-t0:.6f}s")

    t0 = tnow(); ok = test_jacobian_autodiff(L=4, width=64, layers=1, steps=1, device=device,
                                             pretrain_steps=args.pretrain_steps,
                                             pretrain_batch=args.pretrain_batch,
                                             pretrain_lr=args.pretrain_lr); t1 = tnow()
    ok_all &= ok; print(f"  Time: {t1-t0:.6f}s")

    t0 = tnow(); ok = test_sampling(L=args.L, batch=args.batch, device=device); t1 = tnow()
    ok_all &= ok; print(f"  Time: {t1-t0:.6f}s")

    if not args.skip_train:
        t0 = tnow(); ok = tiny_train_step(device=device, L=args.L, stages=args.stages,
                                          batch=args.batch, superbatch=2, width=args.width, layers=args.layers); t1 = tnow()
        ok_all &= ok; print(f"  Time: {t1-t0:.6f}s")

    print("\n=== SUMMARY ===")
    print(f"All tests passed: {ok_all}")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
