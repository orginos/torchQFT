#!/bin/sh

#SBATCH --job-name=MCMG_setup
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log


if [ $# -ne 10 ]; then
    echo "Usage: $0 <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <m_start> <m_end> <m_step>"
    echo "Example: $0 1 2.4 10 8 -1 1000 10000 -0.58 -0.54 0.005"
    exit 1
fi

if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm01,fm04,fm08,fm24"
else
    EXCLUDE=""
fi

Nskip=($1)
lam=($2)
batch=($3)
Lattice=($4)
device=($5)
warm=($6)
meassurements=($7)
m_start=$8
m_end=$9
m_step=${10}

JOB_DIR="/sciclone/pscr/yacahuanamedra/torchQFT"
#add line to ask for gpu if device is -1
if [ $device -eq -1 ]; then
    extraline="#SBATCH --gres=gpu:1"
else
    extraline=""
fi

for m in $(seq $m_start $m_step $m_end); do

    SLURM_SCRIPT="${JOB_DIR}/job_MCMG_${m}.slurm"

    cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=${Nskip}_${lam}_${batch}_${Lattice}_${device}_${warm}_${meassurements}_${m}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/phi4data/specs_data_${m}.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/phi4data/specs_data_${m}.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=8
${extraline}
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --exclude=${EXCLUDE}

cd /sciclone/pscr/yacahuanamedra/torchQFT/


source ~/.bashrc
module load miniforge3/24.9.2-0
conda init
sleep 5
conda activate torchQFT-env

echo "Python path: \$(which python3)"
python3 --version

echo "Running on node: \$(hostname)"
echo "Running on CPUs: \$(scontrol show hostnames \$SLURM_NODELIST)"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "In directory: $(pwd)"


echo "Starting Python at $(date)"
python3 run_MCMG.py -Nskip $Nskip -lam $lam -batch $batch -L $Lattice -dev $device -warm $warm -meas $meassurements -m $m
echo "Finished Python at $(date)"
EOF
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "Job $JOB_ID submitted successfully."
            echo "Running with mass = $m"
            rm -f "$SLURM_SCRIPT"

            echo "SLURM script $SLURM_SCRIPT deleted."
        else
            echo "Failed to submit job. SLURM script not deleted: $SLURM_SCRIPT"
        fi
done
