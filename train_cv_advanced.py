#!/usr/bin/env python3
"""
Advanced Control Variate Training for Phi^4 Theory

This script implements state-of-the-art learning schedules with:
- OneCycleLR policy for smooth learning rate scheduling
- Gradient accumulation for effective larger batch sizes
- Continuous training without reinitialization
- Comprehensive logging and checkpointing

Trains control variates for all tau values from 0 to L/2.
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
from fast_functionals_v2 import fast_model_factory_v2 as fast_model_factory, ALL_MODELS_V2 as ALL_MODELS
import phi4 as qft
import integrators as integ
import update as upd
import tqdm


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
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


def train_with_accumulation(CM, phi, hmc, optimizer, scheduler, epochs, Nskip,
                            accumulation_steps, logger, phase_name, grad_clip=1.0):
    """
    Train with gradient accumulation for effective larger batch sizes.

    Args:
        CM: Control model
        phi: Field configuration
        hmc: HMC updater
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of epochs
        Nskip: HMC steps between samples
        accumulation_steps: Number of gradient accumulation steps
        logger: Logger instance
        phase_name: Name for progress bar
        grad_clip: Gradient clipping norm

    Returns:
        loss_history, lr_history, phi
    """
    loss_history = []
    lr_history = []

    # Note: effective batch size is passed via batch_size argument, not extracted from model
    # logger.log(f"  Effective batch size with accumulation: {accumulation_steps}x")

    pbar = tqdm.tqdm(range(epochs), desc=phase_name)

    for epoch in pbar:
        optimizer.zero_grad()
        accumulated_loss = 0.0

        # Gradient accumulation loop
        for acc_step in range(accumulation_steps):
            # Generate new configurations (keep phi continuous)
            phi = hmc.evolve(phi, Nskip)
            x = phi.clone()
            x.requires_grad = True

            # Compute loss (scaled by accumulation steps)
            loss = CM.loss(x) / accumulation_steps
            loss.backward()

            accumulated_loss += loss.item() * accumulation_steps

        # Gradient clipping
        if grad_clip > 0:
            tr.nn.utils.clip_grad_norm_(CM.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # Record history (average over accumulation steps)
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


def evaluate_model(CM, phi, hmc, Nskip, logger, n_colors_eval=None):
    """Evaluate trained model and log results.

    Args:
        CM: ControlModel
        phi: Field configuration
        hmc: HMC sampler
        Nskip: HMC steps between samples
        logger: Logger
        n_colors_eval: Number of colors for probing during evaluation
                       (use more than training for more accurate Laplacian)
    """

    # Generate fresh configurations for evaluation
    phi = hmc.evolve(phi, Nskip)
    x = phi.clone()
    x.requires_grad = True  # Required for grad_and_lapl in CM.F()

    # Note: Cannot use no_grad() because CM.Delta calls CM.F which uses autograd
    var_O = CM.computeO(x).var().cpu().detach().numpy().item()

    # Use n_colors_eval for more accurate evaluation if specified
    delta = CM.Delta(x, n_colors=n_colors_eval)
    var_impO = delta.var().cpu().detach().numpy().item()
    gain = var_O / var_impO if var_impO > 0 else float('inf')

    muO = CM.muO.cpu().detach().numpy().item()
    mean_O = CM.computeO(x).mean().cpu().detach().numpy().item()
    mean_impO = delta.mean().cpu().detach().numpy().item()

    tF = CM.F(x, n_colors=n_colors_eval).cpu().detach().numpy()
    tO = CM.computeO(x).cpu().detach().numpy()
    corr = np.corrcoef(tF, tO)[0, 1]

    results = {
        'variance_O': var_O,
        'variance_impO': var_impO,
        'variance_gain': gain,
        'muO': muO,
        'mean_O': mean_O,
        'mean_impO': mean_impO,
        'correlation_F_O': corr,
        'n_colors_eval': n_colors_eval
    }

    if n_colors_eval:
        logger.log(f"  Using n_colors={n_colors_eval} for evaluation")
    logger.log(f"  Variance of O: {var_O:.6f}")
    logger.log(f"  Variance of imp(O): {var_impO:.6f}")
    logger.log(f"  Variance improvement: {gain:.2f}x")
    logger.log(f"  muO: {muO:.6f}")
    logger.log(f"  Mean(O): {mean_O:.6f}")
    logger.log(f"  Mean(impO): {mean_impO:.6f}")
    logger.log(f"  Correlation(F,O): {corr:.4f}")

    return results, phi


def create_training_figure(all_phases, tau, output_path):
    """Create and save comprehensive training figure."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Control Variate Training - tau={tau}', fontsize=14)

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

    # Plot 1: Loss history with phase markers
    ax1 = axes[0, 0]
    ax1.plot(epochs, all_losses, 'b-', linewidth=0.8, alpha=0.7)

    # Add phase boundary lines
    colors = ['green', 'orange', 'red', 'purple']
    for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        if i < len(all_phases):
            ax1.axvspan(start, end, alpha=0.1, color=colors[i % len(colors)],
                       label=all_phases[i]['name'])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate schedule
    ax2 = axes[0, 1]
    ax2.plot(epochs, all_lrs, 'r-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.set_title('Learning Rate Schedule (OneCycleLR)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss improvement per phase
    ax3 = axes[1, 0]
    phase_names = [p['name'] for p in all_phases]
    initial_losses = [p['loss'][0] for p in all_phases]
    final_losses = [p['loss'][-1] for p in all_phases]

    x_pos = np.arange(len(phase_names))
    width = 0.35
    ax3.bar(x_pos - width/2, initial_losses, width, label='Initial', color='lightcoral')
    ax3.bar(x_pos + width/2, final_losses, width, label='Final', color='steelblue')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(phase_names, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    ax3.set_title('Loss: Initial vs Final per Phase')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Smoothed loss
    ax4 = axes[1, 1]
    window = min(50, len(all_losses) // 10)
    if window > 1:
        smoothed = np.convolve(all_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = np.arange(window//2, window//2 + len(smoothed))
        ax4.plot(smooth_epochs, smoothed, 'b-', linewidth=1.5)
    else:
        ax4.plot(epochs, all_losses, 'b-', linewidth=1.5)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (smoothed)')
    ax4.set_yscale('log')
    ax4.set_title(f'Smoothed Loss (window={window})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_single_tau(tau, args, logger, device):
    """
    Train control variate for a single tau value with modern training strategy.

    Training Strategy:
    - All phases: OneCycleLR (warmup -> peak -> anneal)
    - Later phases: gradient accumulation for larger effective batch size

    Key: No reinitialization of phi between phases - continuous thermalization
    """

    logger.log(f"\n{'='*60}")
    logger.log(f"Training tau = {tau}")
    logger.log(f"{'='*60}")

    L = args.L
    lat = [L, L]
    batch_size = args.batch_size

    # Create QFT system (only once)
    sg = qft.phi4(lat, args.g, args.m, batch_size=batch_size, device=device)

    # Initialize field configuration
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

    # Create model (supports both original and fast functionals)
    funct = fast_model_factory(args.model, L=L, y=tau,
                               conv_layers=args.conv_l,
                               activation=args.activ,
                               dtype=tr.float,
                               n_colors=args.n_colors,
                               probing_method=args.probing_method)
    funct.to(device)

    # Compute initial muO
    muO = C2pt(phi, tau).mean().cpu().numpy().item()

    # Create control model
    CM = ControlModel(muO=muO, force=sg.force, c2p_net=funct)
    CM.to(device)

    # Count parameters
    param_count = sum(p.numel() for p in CM.parameters() if p.requires_grad)
    logger.log(f"Model parameter count: {param_count}")

    all_phases = []
    params = [p for p in CM.parameters() if p.requires_grad]

    # Create optimizer ONCE - keep state across phases
    optimizer = tr.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    # =========================================================================
    # Training Loop: n_phases stages with increasing batch size
    # All phases: OneCycleLR (warmup -> peak -> anneal)
    # LR scaling: lr / sqrt(accumulation) - less aggressive than 1/accumulation
    # =========================================================================
    for phase_idx in range(args.n_phases):
        accumulation_steps = 2 ** phase_idx  # 1, 2, 4, 8, ...
        # Less aggressive LR reduction: sqrt scaling instead of linear
        #phase_max_lr = args.lr / np.sqrt(accumulation_steps)
        phase_max_lr = args.lr / accumulation_steps
        phase_name = f"Phase{phase_idx + 1}_Batch{accumulation_steps}x"

        logger.log(f"\n--- {phase_name} ---")
        logger.log(f"  Epochs: {args.epochs}, Max LR: {phase_max_lr:.2e}")
        logger.log(f"  Effective batch: {batch_size * accumulation_steps}")

        # Update optimizer LR (keep momentum/Adam state)
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
        loss_hist, lr_hist, phi = train_with_accumulation(
            CM, phi, hmc, optimizer, scheduler,
            args.epochs, args.Nskip,
            accumulation_steps=accumulation_steps,
            logger=logger,
            phase_name=phase_name,
            grad_clip=args.grad_clip
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
    logger.log(f"\n--- Final Evaluation (tau={tau}) ---")
    # Use more colors for evaluation (default: L*L for exact Laplacian)
    n_colors_eval = args.n_colors_eval if args.n_colors_eval else L * L
    eval_results, phi = evaluate_model(CM, phi, hmc, args.Nskip, logger, n_colors_eval=n_colors_eval)

    # Symmetry check
    logger.log(f"\n--- Symmetry Check ---")
    x = phi.clone()
    x.requires_grad = True
    symmetry_checker(x, funct)

    # =========================================================================
    # Save outputs
    # =========================================================================
    output_dir = Path(args.output_dir)
    run_name = f"cv_L_{L}_m_{args.m}_g_{args.g}_tau_{tau}_{args.model}"

    # Save model
    model_path = output_dir / f"{run_name}.dict"
    tr.save(funct.state_dict(), model_path)
    logger.log(f"\nModel saved: {model_path}")

    # Save loss data
    all_losses = np.concatenate([np.array(p['loss']) for p in all_phases])
    all_lrs = np.concatenate([np.array(p['lr_history']) for p in all_phases])

    loss_data = {
        'tau': tau,
        'phases': [{
            'name': p['name'],
            'accumulation': p['accumulation'],
            'initial_loss': p['loss'][0],
            'final_loss': p['loss'][-1],
            'best_loss': min(p['loss']),
            'epochs': len(p['loss'])
        } for p in all_phases],
        'eval_results': eval_results,
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
    create_training_figure(all_phases, tau, fig_path)
    logger.log(f"Training figure saved: {fig_path}")

    return {
        'tau': tau,
        'model_path': str(model_path),
        'final_loss': all_phases[-1]['loss'][-1],
        'variance_gain': eval_results['variance_gain'],
        'correlation': eval_results['correlation_F_O']
    }


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Control Variate Training for Phi^4 Theory'
    )

    # Physics parameters
    parser.add_argument('-L', type=int, default=8, help='Lattice size')
    parser.add_argument('-g', type=float, default=2.4, help='Coupling constant (lambda)')
    parser.add_argument('-m', type=float, default=-0.4, help='Mass parameter')

    # Model parameters
    parser.add_argument('-model', default='Funct3T',
                       help='Model: Funct3T, Funct3T_Hutch, FunctFourier, FunctLocal, FunctQuadratic')
    parser.add_argument('-activ', default='gelu', help='Activation function')
    parser.add_argument('-conv_l', type=int, nargs='+', default=[4, 4, 4, 4],
                       help='Convolutional layer widths')
    parser.add_argument('-n_colors', type=int, default=4,
                       help='Number of probes (coloring: 2,4,9,...; sites: any int up to L²)')
    parser.add_argument('-n_colors_eval', type=int, default=None,
                       help='Number of probes for evaluation (default: L*L for exact)')
    parser.add_argument('-probing_method', default='coloring', choices=['coloring', 'sites'],
                       help='Probing method: coloring (graph coloring) or sites (random site sampling)')

    # Training parameters
    parser.add_argument('-epochs', type=int, default=1000,
                       help='Total training epochs')
    parser.add_argument('-lr', type=float, default=1e-2, help='Max learning rate')
    parser.add_argument('-batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('-grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    parser.add_argument('-n_phases', type=int, default=1,
                       help='Number of training phases (each doubles batch size)')

    # HMC parameters
    parser.add_argument('-Nwarm', type=int, default=1000, help='HMC warmup steps')
    parser.add_argument('-Nskip', type=int, default=5, help='HMC steps between samples')
    parser.add_argument('-Nmd', type=int, default=2, help='MD integration steps')

    # Output
    parser.add_argument('-output_dir', default='trained_models',
                       help='Output directory for models and logs')

    # Tau range (optional override)
    parser.add_argument('-tau_min', type=int, default=None,
                       help='Minimum tau (default: 0)')
    parser.add_argument('-tau_max', type=int, default=None,
                       help='Maximum tau (default: L/2 )')
    parser.add_argument('-tau', type=int, default=None,
                       help='Train single tau value only')

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

    # Determine tau range
    if args.tau is not None:
        tau_min = args.tau
        tau_max = args.tau
    else:
        tau_min = args.tau_min if args.tau_min is not None else 0
        tau_max = args.tau_max if args.tau_max is not None else (args.L // 2)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"L{args.L}_g{args.g}_m{args.m}_{timestamp}"
    logger = TrainingLogger(output_dir, f"training_{run_id}")

    # Log configuration
    logger.log("="*60)
    logger.log("Advanced Control Variate Training")
    logger.log("="*60)
    logger.log(f"Device: {device}")
    logger.log(f"Configuration:")
    logger.log(f"  L={args.L}, g={args.g}, m={args.m}")
    logger.log(f"  Model: {args.model}, Activation: {args.activ}")
    logger.log(f"  Conv layers: {args.conv_l}")
    logger.log(f"  Base epochs: {args.epochs}, Max LR: {args.lr}")
    logger.log(f"  Base batch size: {args.batch_size}")
    logger.log(f"  Tau range: {tau_min} to {tau_max}")
    logger.log(f"  Probing: {args.probing_method} with n_colors={args.n_colors}")
    logger.log(f"  Output: {output_dir}")
    logger.log("="*60)

    logger.history['config'] = vars(args)

    # Train for all tau values
    all_results = []

    for tau in range(tau_min, tau_max + 1):
        try:
            result = train_single_tau(tau, args, logger, device)
            all_results.append(result)
        except Exception as e:
            logger.log(f"ERROR training tau={tau}: {e}")
            import traceback
            logger.log(traceback.format_exc())
            continue

    # Summary
    logger.log("\n" + "="*60)
    logger.log("TRAINING COMPLETE - SUMMARY")
    logger.log("="*60)

    for result in all_results:
        logger.log(f"tau={result['tau']}: gain={result['variance_gain']:.2f}x, "
                  f"corr={result['correlation']:.4f}, loss={result['final_loss']:.6f}")

    # Save summary
    logger.history['results'] = all_results
    logger.save_history()

    # Create summary figure
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        taus = [r['tau'] for r in all_results]
        gains = [r['variance_gain'] for r in all_results]
        corrs = [r['correlation'] for r in all_results]
        losses = [r['final_loss'] for r in all_results]

        axes[0].bar(taus, gains, color='steelblue')
        axes[0].set_xlabel('tau')
        axes[0].set_ylabel('Variance Gain')
        axes[0].set_title('Variance Reduction by tau')
        axes[0].set_xticks(taus)

        axes[1].bar(taus, corrs, color='forestgreen')
        axes[1].set_xlabel('tau')
        axes[1].set_ylabel('Correlation(F, O)')
        axes[1].set_title('Control-Observable Correlation')
        axes[1].set_xticks(taus)

        axes[2].bar(taus, losses, color='coral')
        axes[2].set_xlabel('tau')
        axes[2].set_ylabel('Final Loss')
        axes[2].set_yscale('log')
        axes[2].set_title('Final Training Loss')
        axes[2].set_xticks(taus)

        plt.tight_layout()
        summary_path = output_dir / f"training_summary_{run_id}.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.log(f"\nSummary figure saved: {summary_path}")

    logger.log("\nDone!")


if __name__ == "__main__":
    main()
