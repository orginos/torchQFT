#!/usr/bin/env python3
"""
Benchmark Probing Methods for Control Variates

This script systematically benchmarks different models and probing configurations:
- Models: FunctSmeared2pt, FunctConvSeparable, Funct3T_Probing, FunctSeparable
- Probing method: sites (Karniadakis random site sampling)
- Number of probes: 8, 16, 32, 64

Results are organized in a directory structure:
    benchmark_cv_results/
        <timestamp>_benchmark/
            summary.json           # Overall summary
            summary_table.txt      # Human-readable summary table
            FunctSmeared2pt/
                sites_8/
                sites_16/
                ...
            FunctConvSeparable/
                sites_8/
                ...
            ...
"""

import subprocess
import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import shutil


def run_training(model, n_colors, probing_method, base_dir, args):
    """Run training for a specific configuration."""

    output_dir = base_dir / model / f"{probing_method}_{n_colors}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train_cv_advanced.py",
        "-L", str(args.L),
        "-g", str(args.g),
        "-m", str(args.m_param),
        "-model", model,
        "-n_colors", str(n_colors),
        "-probing_method", probing_method,
        "-epochs", str(args.epochs),
        "-lr", str(args.lr),
        "-batch_size", str(args.batch_size),
        "-n_phases", str(args.n_phases),
        "-Nwarm", str(args.Nwarm),
        "-Nskip", str(args.Nskip),
        "-output_dir", str(output_dir),
        "-device", args.device,
    ]

    # Add tau range if specified
    if args.tau_min is not None:
        cmd.extend(["-tau_min", str(args.tau_min)])
    if args.tau_max is not None:
        cmd.extend(["-tau_max", str(args.tau_max)])
    if args.tau is not None:
        cmd.extend(["-tau", str(args.tau)])

    # Add conv layers for models that use them
    if model in ['FunctConvSeparable', 'FunctSmeared2pt', 'Funct3T_Probing']:
        cmd.extend(["-conv_l"] + [str(c) for c in args.conv_l])

    print(f"\n{'='*70}")
    print(f"Running: {model} with {probing_method}_{n_colors}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run training
    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent),
        capture_output=not args.verbose,
        text=True
    )
    elapsed = time.time() - start_time

    # Check for success
    success = result.returncode == 0

    if not success and not args.verbose:
        print(f"FAILED! Return code: {result.returncode}")
        print("STDOUT:", result.stdout[-2000:] if result.stdout else "None")
        print("STDERR:", result.stderr[-2000:] if result.stderr else "None")

    # Try to load metrics
    metrics = None
    metrics_file = output_dir / "all_tau_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except:
            pass

    return {
        'model': model,
        'probing_method': probing_method,
        'n_colors': n_colors,
        'success': success,
        'elapsed_seconds': elapsed,
        'output_dir': str(output_dir),
        'metrics': metrics
    }


def create_summary_table(results, output_file):
    """Create a human-readable summary table."""

    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("BENCHMARK SUMMARY: Probing Methods for Control Variates\n")
        f.write("=" * 100 + "\n\n")

        # Group by model
        models = sorted(set(r['model'] for r in results))
        n_colors_list = sorted(set(r['n_colors'] for r in results))

        # Header
        f.write(f"{'Model':<25} {'n_colors':<10} {'Status':<10} {'Time (s)':<12} {'Avg Gain':<12} {'Min Gain':<12}\n")
        f.write("-" * 100 + "\n")

        for model in models:
            model_results = [r for r in results if r['model'] == model]
            for r in sorted(model_results, key=lambda x: x['n_colors']):
                status = "OK" if r['success'] else "FAILED"
                time_str = f"{r['elapsed_seconds']:.1f}"

                # Extract gains from metrics
                avg_gain = "N/A"
                min_gain = "N/A"
                if r['metrics']:
                    gains = []
                    for tau_key, tau_data in r['metrics'].items():
                        if tau_key.startswith('tau_') and 'variance_gain' in tau_data:
                            gains.append(tau_data['variance_gain'])
                    if gains:
                        avg_gain = f"{sum(gains)/len(gains):.2f}x"
                        min_gain = f"{min(gains):.2f}x"

                f.write(f"{model:<25} {r['n_colors']:<10} {status:<10} {time_str:<12} {avg_gain:<12} {min_gain:<12}\n")
            f.write("-" * 100 + "\n")

        # Detailed results per tau
        f.write("\n\nDETAILED RESULTS PER TAU:\n")
        f.write("=" * 100 + "\n\n")

        for r in results:
            if not r['metrics']:
                continue

            f.write(f"\n{r['model']} with {r['probing_method']}_{r['n_colors']}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'tau':<6} {'var_orig':<15} {'var_imp':<15} {'gain':<10}\n")

            for tau in range(20):  # Reasonable max tau
                tau_key = f"tau_{tau}"
                if tau_key in r['metrics']:
                    data = r['metrics'][tau_key]
                    var_orig = data.get('var_original', 0)
                    var_imp = data.get('var_improved', 0)
                    gain = data.get('variance_gain', 0)
                    f.write(f"{tau:<6} {var_orig:<15.6f} {var_imp:<15.6f} {gain:<10.2f}x\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark probing methods for control variates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Models to test
    parser.add_argument('-models', nargs='+',
                        default=['FunctSmeared2pt', 'FunctConvSeparable', 'Funct3T_Probing', 'FunctSeparable'],
                        help='Models to benchmark')
    parser.add_argument('-n_colors_list', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='Number of probing colors/sites to test')
    parser.add_argument('-probing_method', default='sites', choices=['coloring', 'sites'],
                        help='Probing method')

    # Physics parameters
    parser.add_argument('-L', type=int, default=8, help='Lattice size')
    parser.add_argument('-g', type=float, default=2.4, help='Coupling constant')
    parser.add_argument('-m_param', type=float, default=-0.4, help='Mass parameter')

    # Training parameters
    parser.add_argument('-epochs', type=int, default=500, help='Epochs per phase')
    parser.add_argument('-lr', type=float, default=1e-2, help='Max learning rate')
    parser.add_argument('-batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('-n_phases', type=int, default=1, help='Training phases')
    parser.add_argument('-conv_l', type=int, nargs='+', default=[4, 4, 4, 4],
                        help='Conv layer channels')

    # HMC parameters
    parser.add_argument('-Nwarm', type=int, default=500, help='HMC warmup steps')
    parser.add_argument('-Nskip', type=int, default=5, help='HMC skip steps')

    # Tau range
    parser.add_argument('-tau_min', type=int, default=None, help='Min tau to train')
    parser.add_argument('-tau_max', type=int, default=None, help='Max tau to train')
    parser.add_argument('-tau', type=int, default=None, help='Single tau to train')

    # Output
    parser.add_argument('-output_dir', default='benchmark_cv_results',
                        help='Base output directory')
    parser.add_argument('-run_name', default=None,
                        help='Run name (default: timestamp)')

    # Execution
    parser.add_argument('-device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('-verbose', action='store_true',
                        help='Show full training output')
    parser.add_argument('-dry_run', action='store_true',
                        help='Just print what would be run')

    args = parser.parse_args()

    # Create output directory
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_benchmark"

    base_dir = Path(args.output_dir) / args.run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBenchmark Configuration:")
    print(f"  Models: {args.models}")
    print(f"  Probing method: {args.probing_method}")
    print(f"  n_colors: {args.n_colors_list}")
    print(f"  Output: {base_dir}")
    print(f"  Total runs: {len(args.models) * len(args.n_colors_list)}")
    print()

    if args.dry_run:
        print("DRY RUN - would run:")
        for model in args.models:
            for n_colors in args.n_colors_list:
                print(f"  {model} with {args.probing_method}_{n_colors}")
        return

    # Save config
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    with open(base_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Run benchmarks
    results = []
    total_runs = len(args.models) * len(args.n_colors_list)
    run_idx = 0

    start_time = time.time()

    for model in args.models:
        for n_colors in args.n_colors_list:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Starting {model} with {args.probing_method}_{n_colors}")

            result = run_training(
                model=model,
                n_colors=n_colors,
                probing_method=args.probing_method,
                base_dir=base_dir,
                args=args
            )
            results.append(result)

            # Save intermediate results
            with open(base_dir / "results_partial.json", 'w') as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    # Save final results
    summary = {
        'config': config,
        'results': results,
        'total_time_seconds': total_time,
        'successful_runs': sum(1 for r in results if r['success']),
        'failed_runs': sum(1 for r in results if not r['success'])
    }

    with open(base_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Create human-readable summary
    create_summary_table(results, base_dir / "summary_table.txt")

    # Print final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {summary['successful_runs']}/{total_runs}")
    print(f"Results saved to: {base_dir}")
    print(f"\nSummary table: {base_dir / 'summary_table.txt'}")

    # Quick summary of gains
    print("\nQuick Summary (Average Gains):")
    print("-" * 50)
    for model in args.models:
        model_results = [r for r in results if r['model'] == model and r['success']]
        if not model_results:
            continue
        print(f"\n{model}:")
        for r in sorted(model_results, key=lambda x: x['n_colors']):
            if r['metrics']:
                gains = []
                for tau_key, tau_data in r['metrics'].items():
                    if tau_key.startswith('tau_') and 'variance_gain' in tau_data:
                        gains.append(tau_data['variance_gain'])
                if gains:
                    avg = sum(gains) / len(gains)
                    print(f"  n_colors={r['n_colors']:>3}: avg gain = {avg:.2f}x")


if __name__ == "__main__":
    main()
