#!/usr/bin/env python3
"""
Unified Control Variate Training for Phi^4 Theory

This script trains a SINGLE Funct3T_Unified model that handles ALL tau values
simultaneously using FiLM (Feature-wise Linear Modulation) conditioning.

Key differences from train_cv_advanced.py:
- ONE model handles all tau ∈ [0, L/2]
- Joint training on all tau values simultaneously
- Each batch samples randomly across tau values
"""

import torch as tr
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Import project modules
from control_variates import (
    C2pt, symmetry_checker, ControlModel,
    model_factory as original_model_factory, activation_factory
)
from fast_functionals_v2 import fast_model_factory_v2 as fast_model_factory, ALL_MODELS_V2 as ALL_MODELS, Funct3T_Unified, Funct3TUnified
import phi4 as qft
import integrators as integ
import update as upd
import tqdm


class UnifiedControlModel(nn.Module):
    """
    Control model wrapper for Funct3T_Unified that handles all tau values.

    Similar to ControlModel but with tau-aware methods.

    Key features:
    - Enforces symmetry: F(tau) = F(L - tau) for tau > L/2
    - Three loss types: 'log', 'gain', 'sum'
    - Caches Var[O] for efficiency in gain loss
    """

    def __init__(self, muO_dict, varO_dict, force, c2p_net, L):
        """
        Args:
            muO_dict: dict mapping tau -> muO value (mean of observable)
            varO_dict: dict mapping tau -> varO value (variance of observable)
            force: QFT force function
            c2p_net: Funct3T_Unified network
            L: lattice size
        """
        super().__init__()
        self.c2p_net = c2p_net
        self.force = force
        self.L = L
        self.tau_max = L // 2 + 1  # tau in [0, L/2]

        # Store muO and varO for each tau as buffers
        muO_tensor = tr.zeros(self.tau_max)
        varO_tensor = tr.zeros(self.tau_max)
        for tau in range(self.tau_max):
            if tau in muO_dict:
                muO_tensor[tau] = muO_dict[tau]
            if tau in varO_dict:
                varO_tensor[tau] = varO_dict[tau]
        self.register_buffer('muO_all', muO_tensor)
        self.register_buffer('varO_all', varO_tensor)

    def _fold_tau(self, tau):
        """Map tau to [0, L/2] using symmetry: tau -> min(tau, L - tau)."""
        if tau > self.L // 2:
            return self.L - tau
        return tau

    def F(self, x, tau, n_colors=None):
        """Compute F = nabla f . nabla phi + Laplacian f for given tau.

        Uses symmetry: F(tau) = F(L - tau) for tau > L/2.
        """
        # Fold tau to [0, L/2]
        tau_folded = self._fold_tau(tau)
        self.c2p_net.set_tau(tau_folded)

        if n_colors is not None and hasattr(self.c2p_net, 'grad_and_lapl'):
            try:
                g, l = self.c2p_net.grad_and_lapl(x, tau=tau_folded, n_colors=n_colors)
            except TypeError:
                g, l = self.c2p_net.grad_and_lapl(x, tau=tau_folded)
        else:
            g, l = self.c2p_net.grad_and_lapl(x, tau=tau_folded)

        return -(g * self.force(x)).sum(dim=(-2, -1)) + l

    def computeO(self, x, tau):
        """Compute the observable C2pt for given tau."""
        return C2pt(x, tau)

    def Delta(self, x, tau, n_colors=None):
        """Compute improved observable O - F + muO for given tau."""
        O = self.computeO(x, tau)
        F = self.F(x, tau, n_colors=n_colors)
        return O - F 

    
    def var_delta(self, x, tau, n_colors=None):
        """Compute Var(Delta) for given tau."""
        tau_folded = self._fold_tau(tau)
        muO = self.muO_all[tau_folded]
        delta = self.Delta(x, tau, n_colors=n_colors)
        # I cannot be taking the variance here because I loose
        # sensitivity to muO in the loss function
#        return delta.var()
        return ((delta-muO)**2).mean()

    def multi_tau_loss(self, x, tau_list, n_colors=None, loss_type='log'):
        """
        Compute loss over multiple tau values.

        Args:
            x: Field configuration (batch, L, L)
            tau_list: List of tau values to include (should be in [0, L/2])
            n_colors: Number of colors for probing
            loss_type: 'log' (default), 'gain', or 'sum'
                - 'log': Σ log(Var[Δ]) - scale invariant, maximizes geometric mean of gains
                - 'gain': Σ Var[Δ]/Var[O] = Σ 1/gain - minimizes inverse gains
                - 'sum': Σ Var[Δ] - simple sum (dominated by small tau)

        Returns:
            Loss value (scalar tensor)
        """
        total_loss = 0.0
        n_tau = len(tau_list)

        for tau in tau_list:
            tau_folded = self._fold_tau(tau)
            var_delta = self.var_delta(x, tau, n_colors=n_colors)

            if loss_type == 'log':
                # Log loss: scale-invariant, balances all tau
                total_loss = total_loss + tr.log(var_delta + 1e-10)
            elif loss_type == 'gain':
                # Gain loss: Var[Δ] / Var[O] = 1/gain
                var_O = self.varO_all[tau_folded]
                total_loss = total_loss + var_delta / (var_O + 1e-10)
            else:  # 'sum'
                # Simple sum (original behavior)
                total_loss = total_loss + var_delta

        return total_loss / n_tau


def train_unified(UCM, phi, hmc, optimizer, scheduler, epochs, Nskip, tau_list,
                  accumulation_steps, logger, phase_name, grad_clip=1.0, n_colors=None,
                  loss_type='log'):
    """
    Train unified model on all tau values simultaneously.

    Each epoch:
    1. Sample new configurations
    2. Compute loss for ALL tau values (using specified loss_type)
    3. Average losses and backprop

    Args:
        loss_type: 'log' (default), 'gain', or 'sum'
    """
    loss_history = []
    lr_history = []

    pbar = tqdm.tqdm(range(epochs), desc=phase_name)

    for epoch in pbar:
        optimizer.zero_grad()
        accumulated_loss = 0.0

        # Gradient accumulation loop
        for acc_step in range(accumulation_steps):
            # Generate new configurations
            phi = hmc.evolve(phi, Nskip)
            x = phi.clone()
            x.requires_grad = True

            # Compute loss over ALL tau values (scaled by accumulation)
            loss = UCM.multi_tau_loss(x, tau_list, n_colors=n_colors, loss_type=loss_type) / accumulation_steps
            loss.backward()

            accumulated_loss += loss.item() * accumulation_steps

        # Gradient clipping
        if grad_clip > 0:
            tr.nn.utils.clip_grad_norm_(UCM.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # Record history
        loss_history.append(accumulated_loss / accumulation_steps)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Update progress bar
        avg_loss = accumulated_loss / accumulation_steps
        pbar.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'lr': f'{current_lr:.2e}'
        })

    return loss_history, lr_history, phi


def evaluate_unified(UCM, phi, hmc, Nskip, tau_list, logger, n_colors_eval=None):
    """Evaluate unified model on all tau values."""
    results = {}

    # Generate fresh configurations
    phi = hmc.evolve(phi, Nskip)
    x = phi.clone()
    x.requires_grad = True

    for tau in tau_list:
        var_O = UCM.computeO(x, tau).var().cpu().detach().numpy().item()
        delta = UCM.Delta(x, tau, n_colors=n_colors_eval)
        var_impO = delta.var().cpu().detach().numpy().item()
        gain = var_O / var_impO if var_impO > 0 else float('inf')

        muO = UCM.muO_all[tau].cpu().detach().numpy().item()
        mean_O = UCM.computeO(x, tau).mean().cpu().detach().numpy().item()
        mean_impO = delta.mean().cpu().detach().numpy().item()

        tF = UCM.F(x, tau, n_colors=n_colors_eval).cpu().detach().numpy()
        tO = UCM.computeO(x, tau).cpu().detach().numpy()
        corr = np.corrcoef(tF, tO)[0, 1]

        results[tau] = {
            'variance_O': var_O,
            'variance_impO': var_impO,
            'variance_gain': gain,
            'muO': muO,
            'mean_O': mean_O,
            'mean_impO': mean_impO,
            'correlation_F_O': corr
        }

        logger.log(f"  tau={tau}: Gain={gain:.2f}x, Corr={corr:.4f}, Var(O)={var_O:.6f}")

    # Average gain across all tau
    avg_gain = np.mean([r['variance_gain'] for r in results.values()])
    logger.log(f"  Average gain across all tau: {avg_gain:.2f}x")

    return results, phi


def create_unified_training_figure(all_phases, tau_list, eval_results, output_path, loss_type='log'):
    """Create and save comprehensive training figure for unified model."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Unified Control Variate Training - tau ∈ {tau_list} (loss: {loss_type})', fontsize=14)

    # Concatenate all data
    all_losses = []
    all_lrs = []
    phase_boundaries = [0]

    for phase in all_phases:
        all_losses.extend(phase['loss'])
        all_lrs.extend(phase['lr_history'])
        phase_boundaries.append(len(all_losses))

    all_losses = np.array(all_losses)
    all_lrs = np.array(all_lrs)
    epochs = np.arange(len(all_losses))

    # Plot 1: Loss history
    ax1 = axes[0, 0]
    ax1.plot(epochs, all_losses, 'b-', linewidth=0.8, alpha=0.7)
    colors = ['green', 'orange', 'red', 'purple']
    for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        if i < len(all_phases):
            ax1.axvspan(start, end, alpha=0.1, color=colors[i % len(colors)],
                       label=all_phases[i]['name'])
    ax1.set_xlabel('Epoch')
    # For log loss, values are already log-scaled (negative), use linear
    # For sum/gain loss, use log scale
    if loss_type == 'log':
        ax1.set_ylabel('Log Loss (= Σ log Var[Δ])')
        ax1.set_title('Training Loss (log-scale, lower is better)')
    else:
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.set_title('Training Loss (averaged over all tau)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate schedule
    ax2 = axes[0, 1]
    ax2.plot(epochs, all_lrs, 'r-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Variance gain per tau
    ax3 = axes[1, 0]
    taus = sorted(eval_results.keys())
    gains = [eval_results[tau]['variance_gain'] for tau in taus]
    ax3.bar(taus, gains, color='steelblue', edgecolor='black')
    ax3.set_xlabel('tau')
    ax3.set_ylabel('Variance Gain')
    ax3.set_title('Variance Improvement per tau')
    ax3.set_xticks(taus)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Correlation per tau
    ax4 = axes[1, 1]
    corrs = [eval_results[tau]['correlation_F_O'] for tau in taus]
    ax4.bar(taus, corrs, color='coral', edgecolor='black')
    ax4.set_xlabel('tau')
    ax4.set_ylabel('Correlation(F, O)')
    ax4.set_title('Correlation per tau')
    ax4.set_xticks(taus)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class TrainingLogger:
    """Comprehensive logging for training runs."""

    def __init__(self, output_dir, run_name):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.log_file = self.output_dir / f"{run_name}_log.txt"
        self.history = {
            'phases': [],
            'config': {},
            'results': {}
        }

    def log(self, message, also_print=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        if also_print:
            print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def save_history(self):
        with open(self.output_dir / f"{self.run_name}_history.json", 'w') as f:
            history_serializable = self._make_serializable(self.history)
            json.dump(history_serializable, f, indent=2)

    def _make_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Unified Control Variate Training for Phi^4 Theory'
    )

    # Physics parameters
    parser.add_argument('-L', type=int, default=8, help='Lattice size')
    parser.add_argument('-g', type=float, default=2.4, help='Coupling constant (lambda)')
    parser.add_argument('-m', type=float, default=-0.4, help='Mass parameter')

    # Model parameters
    parser.add_argument('-model', default='Funct3T_Unified',
                       choices=['Funct3T_Unified', 'Funct3TUnified'],
                       help='Model: Funct3T_Unified (FiLM) or Funct3TUnified (concat)')
    parser.add_argument('-activ', default='gelu', help='Activation function')
    parser.add_argument('-conv_l', type=int, nargs='+', default=[4, 4, 4, 4],
                       help='Convolutional layer widths')
    parser.add_argument('-n_colors', type=int, default=4,
                       help='Number of probes for Laplacian (coloring: 2,4,9,...; random: any int)')
    parser.add_argument('-n_colors_eval', type=int, default=None,
                       help='Number of probes for evaluation (default: L*L for exact)')
    parser.add_argument('-probing_method', default='coloring', choices=['coloring', 'sites'],
                       help='Probing method: coloring (graph coloring) or sites (random site sampling)')

    # Training parameters
    parser.add_argument('-epochs', type=int, default=1000,
                       help='Epochs per phase')
    parser.add_argument('-lr', type=float, default=1e-2, help='Max learning rate')
    parser.add_argument('-batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('-grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    parser.add_argument('-n_phases', type=int, default=1,
                       help='Number of training phases')
    parser.add_argument('-loss_type', default='log', choices=['log', 'gain', 'sum'],
                       help='Loss type: log (default), gain, or sum')

    # HMC parameters
    parser.add_argument('-Nwarm', type=int, default=1000, help='HMC warmup steps')
    parser.add_argument('-Nskip', type=int, default=5, help='HMC steps between samples')
    parser.add_argument('-Nmd', type=int, default=2, help='MD integration steps')

    # Output
    parser.add_argument('-output_dir', default='trained_models',
                       help='Output directory for models and logs')

    # Device
    parser.add_argument('-device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device: auto (default), cpu, cuda, or mps')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if tr.cuda.is_available():
            device = tr.device('cuda')
        elif tr.backends.mps.is_available():
            device = tr.device('mps')
        else:
            device = tr.device('cpu')
    else:
        device = tr.device(args.device)

    L = args.L
    tau_list = list(range(L // 2 + 1))  # All tau from 0 to L/2

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"unified_L{L}_g{args.g}_m{args.m}_{timestamp}"
    logger = TrainingLogger(output_dir, f"training_{run_id}")

    # Log configuration
    logger.log("="*60)
    logger.log("Unified Control Variate Training")
    logger.log("="*60)
    logger.log(f"Device: {device}")
    logger.log(f"Configuration:")
    logger.log(f"  L={L}, g={args.g}, m={args.m}")
    logger.log(f"  Model: {args.model}, Activation: {args.activ}")
    logger.log(f"  Conv layers: {args.conv_l}")
    logger.log(f"  Epochs per phase: {args.epochs}, Max LR: {args.lr}")
    logger.log(f"  Batch size: {args.batch_size}")
    logger.log(f"  N phases: {args.n_phases}")
    logger.log(f"  Loss type: {args.loss_type}")
    logger.log(f"  N colors (train): {args.n_colors}")
    logger.log(f"  Probing method: {args.probing_method}")
    logger.log(f"  Tau values: {tau_list}")
    logger.log(f"  Output: {output_dir}")
    logger.log("="*60)

    logger.history['config'] = vars(args)

    # Create QFT system
    lat = [L, L]
    batch_size = args.batch_size
    sg = qft.phi4(lat, args.g, args.m, batch_size=batch_size, device=device)

    # Initialize field
    phi = sg.hotStart()

    # Setup HMC
    mn2 = integ.minnorm2(sg.force, sg.evolveQ, args.Nmd, 1.0)
    hmc = upd.hmc(T=sg, I=mn2, verbose=False)

    # Initial warmup
    logger.log(f"Warming up HMC ({args.Nwarm} steps)...")
    tic = time.perf_counter()
    phi = hmc.evolve(phi, args.Nwarm)
    toc = time.perf_counter()
    logger.log(f"  HMC warmup: {(toc-tic)*1e3/args.Nwarm:.2f} ms/trajectory")
    logger.log(f"  HMC acceptance: {hmc.calc_Acceptance():.3f}")

    # Compute muO and varO for each tau
    logger.log("Computing muO and varO for each tau...")
    muO_dict = {}
    varO_dict = {}
    for tau in tau_list:
        O_tau = C2pt(phi, tau)
        muO = O_tau.mean().cpu().numpy().item()
        varO = O_tau.var().cpu().numpy().item()
        muO_dict[tau] = muO
        varO_dict[tau] = varO
        logger.log(f"  tau={tau}: muO={muO:.6e}, varO={varO:.6e}")

    # Create unified model
    activ = activation_factory(args.activ)
    if args.model == 'Funct3T_Unified':
        funct = Funct3T_Unified(L=L, conv_layers=args.conv_l, n_colors=args.n_colors,
                                dtype=tr.float32, activation=activ,
                                probing_method=args.probing_method)
    else:  # Funct3TUnified
        funct = Funct3TUnified(L=L, conv_layers=args.conv_l, n_colors=args.n_colors,
                               dtype=tr.float32, activation=activ,
                               probing_method=args.probing_method)
    funct.to(device)

    # Count parameters
    param_count = sum(p.numel() for p in funct.parameters() if p.requires_grad)
    logger.log(f"Model parameter count: {param_count}")

    # Create unified control model
    UCM = UnifiedControlModel(muO_dict=muO_dict, varO_dict=varO_dict,
                              force=sg.force, c2p_net=funct, L=L)
    UCM.to(device)

    all_phases = []
    params = [p for p in UCM.parameters() if p.requires_grad]

    # Create optimizer
    optimizer = tr.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    # =========================================================================
    # Training Loop
    # =========================================================================
    for phase_idx in range(args.n_phases):
        accumulation_steps = 2 ** phase_idx
        #phase_max_lr = args.lr / np.sqrt(accumulation_steps)
        phase_max_lr = args.lr / accumulation_steps
        phase_name = f"Phase{phase_idx + 1}_Batch{accumulation_steps}x"

        logger.log(f"\n--- {phase_name} ---")
        logger.log(f"  Epochs: {args.epochs}, Max LR: {phase_max_lr:.2e}")
        logger.log(f"  Effective batch: {batch_size * accumulation_steps}")

        # Update optimizer LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = phase_max_lr

        # OneCycleLR for all phases - better results
        scheduler = tr.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=phase_max_lr,
            epochs=args.epochs,
            steps_per_epoch=1,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )

        hmc.AcceptReject = []
        loss_hist, lr_hist, phi = train_unified(
            UCM, phi, hmc, optimizer, scheduler,
            args.epochs, args.Nskip, tau_list,
            accumulation_steps=accumulation_steps,
            logger=logger,
            phase_name=phase_name,
            grad_clip=args.grad_clip,
            n_colors=args.n_colors,
            loss_type=args.loss_type
        )

        logger.log(f"  HMC acceptance: {hmc.calc_Acceptance():.3f}")
        logger.log(f"  Initial loss: {loss_hist[0]:.6f}")
        logger.log(f"  Final loss: {loss_hist[-1]:.6f}")
        logger.log(f"  Best loss: {min(loss_hist):.6f}")

        all_phases.append({
            'name': phase_name,
            'loss': loss_hist,
            'lr_history': lr_hist,
            'accumulation': accumulation_steps
        })

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    logger.log(f"\n--- Final Evaluation (all tau) ---")
    n_colors_eval = args.n_colors_eval if args.n_colors_eval else L * L
    logger.log(f"  Using n_colors={n_colors_eval} for evaluation")
    eval_results, phi = evaluate_unified(UCM, phi, hmc, args.Nskip, tau_list, logger,
                                         n_colors_eval=n_colors_eval)

    # =========================================================================
    # Save outputs
    # =========================================================================
    run_name = f"cv_unified_L_{L}_m_{args.m}_g_{args.g}"

    # Save model
    model_path = output_dir / f"{run_name}.dict"
    tr.save(funct.state_dict(), model_path)
    logger.log(f"\nModel saved: {model_path}")

    # Save loss data
    all_losses = np.concatenate([np.array(p['loss']) for p in all_phases])
    all_lrs = np.concatenate([np.array(p['lr_history']) for p in all_phases])

    loss_data = {
        'tau_list': tau_list,
        'phases': [{
            'name': p['name'],
            'accumulation': p['accumulation'],
            'initial_loss': p['loss'][0],
            'final_loss': p['loss'][-1],
            'best_loss': min(p['loss']),
            'epochs': len(p['loss'])
        } for p in all_phases],
        'eval_results': {str(k): v for k, v in eval_results.items()},
        'total_epochs': len(all_losses)
    }

    loss_path = output_dir / f"{run_name}_loss.npz"
    np.savez(loss_path,
             all_losses=all_losses,
             all_lrs=all_lrs,
             loss_data=json.dumps(loss_data))
    logger.log(f"Loss data saved: {loss_path}")

    # Create and save training figure
    fig_path = output_dir / f"{run_name}_training.png"
    create_unified_training_figure(all_phases, tau_list, eval_results, fig_path, loss_type=args.loss_type)
    logger.log(f"Training figure saved: {fig_path}")

    # Save history
    logger.history['results'] = eval_results
    logger.save_history()

    # Summary
    logger.log("\n" + "="*60)
    logger.log("UNIFIED TRAINING COMPLETE")
    logger.log("="*60)
    avg_gain = np.mean([r['variance_gain'] for r in eval_results.values()])
    logger.log(f"Average variance gain: {avg_gain:.2f}x")
    logger.log(f"Model: {model_path}")
    logger.log("="*60)


if __name__ == '__main__':
    main()
