#!/bin/bash
#SBATCH --job-name=1_2.4_10_8_-1_10_100_-0.580
#SBATCH --account=hadron
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/global/homes/y/yacm/pscratch/torchQFT/phi4data_hmc/specs_data_-0.580.log
#SBATCH --error=/global/homes/y/yacm/pscratch/torchQFT/phi4data_hmc/specs_data_-0.580.log

set -euo pipefail

cd "/global/homes/y/yacm/pscratch/torchQFT"

# Load your runtime environment (customize as needed)
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

if command -v module >/dev/null 2>&1; then
    module load python >/dev/null 2>&1 || true
fi

if [ -n "torchqft" ]; then
    conda activate "torchqft"
elif [ -n "${VENV_PATH:-}" ]; then
    source "${VENV_PATH}/bin/activate"
fi

echo "Python path: $(which python3)"
python3 --version
echo "Running on node: $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM job name: $SLURM_JOB_NAME"
echo "Working directory: $(pwd)"

echo "Starting Python at $(date)"
python3 run_MCMG.py -Nskip 1 -lam 2.4 -batch 10 -L 8 -dev -1 -warm 10 -meas 100 -m -0.580 -mcmg hmc -train DeltaS -Nlayers 4 -width 4 -Nlevels 2 -snet yes
echo "Finished Python at $(date)"
