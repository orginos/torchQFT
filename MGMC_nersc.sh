#!/bin/bash

# Submit one Perlmutter job per mass value for run_MCMG.py
# Usage:
#   bash MGMC_nersc.sh <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <m_start> <m_end> <m_step> <mcmg> <train> <Nlayers> <width> <Nlevels> <snet> <block> [time_hours]

set -euo pipefail

if [ $# -ne 17 ] && [ $# -ne 18 ]; then
    echo "Usage: $0 <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <m_start> <m_end> <m_step> <mcmg> <train> <Nlayers> <width> <Nlevels> <snet> <block> [time_hours]"
    echo "Example: $0 ./MGMC_nersc.sh 1 2.4 100 16 cuda 1000 10000 -0.58 -0.55 0.01 hmc DeltaS 2 2 1 yes average 1"
    exit 1
fi

Nskip=$1
lam=$2
batch=$3
Lattice=$4
device=$5
warm=$6
meassurements=$7
m_start=$8
m_end=$9
m_step=${10}
mcmg=${11}
train=${12}
Nlayers=${13}
width=${14}
Nlevels=${15}
snet=${16}
block=${17}
time_hours=${18:-}

#print the parameters for logging
echo "Parameters:"
echo "Nskip: ${Nskip}, lam: ${lam}, batch: ${batch} Lattice: ${Lattice}, device: ${device}, warm: ${warm}, meassurements: ${meassurements}, m_start: ${m_start}, m_end: ${m_end}, m_step: ${m_step}, mcmg: ${mcmg}, train: ${train}, Nlayers: ${Nlayers}, width: ${width}, Nlevels: ${Nlevels}, snet: ${snet}, block: ${block}, time_hours: ${time_hours:-env/default}"

if [ "${m_step}" = "0" ]; then
    echo "Error: m_step cannot be 0."
    exit 1
fi

method_label=$mcmg
if [ "$mcmg" = "rnvp" ]; then
    if [ "$snet" = "yes" ]; then
        method_label="rnvp_flow"
    else
        method_label="deepsets_flow"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_DIR="${SCRIPT_DIR}"
output_dir_name="phi4data_${method_label}"
mkdir -p "${JOB_DIR}/${output_dir_name}"

# NERSC/Perlmutter settings (override with env vars if needed)
# User defaults:
# - project/account: hadron
# - conda env: torchqft
ACCOUNT="${NERSC_ACCOUNT:-${SLURM_ACCOUNT:-hadron}}"

QOS_GPU="${NERSC_QOS_GPU:-regular}"
QOS_CPU="${NERSC_QOS_CPU:-regular}"
if [ -n "${time_hours}" ]; then
    if ! [[ "${time_hours}" =~ ^[0-9]+$ ]] || [ "${time_hours}" -le 0 ]; then
        echo "Error: time_hours must be a positive integer (hours)."
        exit 1
    fi
    WALLTIME="${time_hours}:00:00"
else
    WALLTIME="${NERSC_WALLTIME:-1:00:00}"
fi
CONDA_ENV_DEFAULT="${CONDA_ENV_NAME:-torchqft}"

# device names:
# - "cuda" => request GPU
# - anything else => CPU
if [ "$device" = "cuda" ]; then
    constraint="gpu"
    qos="${QOS_GPU}"
    extra_resources=$'#SBATCH --gpus=1\n#SBATCH --cpus-per-task=2'
else
    constraint="cpu"
    qos="${QOS_CPU}"
    extra_resources="#SBATCH --cpus-per-task=4"
fi

for m in $(seq "$m_start" "$m_step" "$m_end"); do
    SLURM_SCRIPT="${JOB_DIR}/job_MGMC_${m}.slurm"

    cat > "${SLURM_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${mcmg}_${Lattice}_${device}_${warm}_${meassurements}_${m}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${qos}
#SBATCH --constraint=${constraint}
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
${extra_resources}
#SBATCH --output=${JOB_DIR}/${output_dir_name}/L_${Lattice}_specs_data_${m}.log
#SBATCH --error=${JOB_DIR}/${output_dir_name}/L_${Lattice}_specs_data_${m}.log

set -euo pipefail

cd "${JOB_DIR}"

# Load your runtime environment (customize as needed)
if [ -f "\$HOME/.bashrc" ]; then
    source "\$HOME/.bashrc"
fi

if command -v module >/dev/null 2>&1; then
    module load python >/dev/null 2>&1 || true
fi

if [ -n "${CONDA_ENV_DEFAULT}" ]; then
    conda activate "${CONDA_ENV_DEFAULT}"
elif [ -n "\${VENV_PATH:-}" ]; then
    source "\${VENV_PATH}/bin/activate"
fi

echo "Python path: \$(which python3)"
python3 --version
echo "Running on node: \$(hostname)"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "Working directory: \$(pwd)"

echo "Starting Python at \$(date)"
python3 run_MCMG.py -Nskip ${Nskip} -lam ${lam} -batch ${batch} -L ${Lattice} -dev ${device} -warm ${warm} -meas ${meassurements} -m ${m} -mcmg ${mcmg} -train ${train} -Nlayers ${Nlayers} -width ${width} -Nlevels ${Nlevels} -snet ${snet}
echo "Finished Python at \$(date)"
EOF

    JOB_ID=$(sbatch "${SLURM_SCRIPT}" | awk '{print $4}')
    if [ $? -eq 0 ]; then
        echo "Job ${JOB_ID} submitted successfully (mass=${m})."
        rm -f "${SLURM_SCRIPT}"
    else
        echo "Failed to submit job for mass=${m}. Script kept at ${SLURM_SCRIPT}"
    fi
done
