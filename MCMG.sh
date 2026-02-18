#!/bin/sh

#SBATCH --job-name=MCMG_setup
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log


if [ $# -ne 16 ]; then
    echo "Usage: $0 <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <m_start> <m_end> <m_step> <mcmg> <train> <Nlayers> <width> <Nlevels> <snet>"
    echo "Example: $0 1 2.4 10 8 -1 1000 10000 -0.55 -0.58 -0.005 hmc none 4 4 2 yes"
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
mcmg=${11}
train=${12}
Nlayers=${13}
width=${14}
Nlevels=${15}
snet=${16}

method_label=$mcmg
if [ "$mcmg" = "rnvp" ]; then
    if [ "$snet" = "yes" ]; then
        method_label="rnvp_flow"
    else
        method_label="deepsets_flow"
    fi
fi

output_dir_name="phi4data_${method_label}"


JOB_DIR="/sciclone/pscr/yacahuanamedra/torchQFT"
#add line to ask for gpu if device is -1
if [ $device -eq -1 ]; then
    extraline="#SBATCH --gres=gpu:1" # Solicita una GPU.
    cpus_per_task_val=4 # Núcleos de CPU para soportar la GPU.
else
    extraline=""
    cpus_per_task_val=8 # Más núcleos de CPU para ejecución solo en CPU.
    # Considera aumentar --mem para trabajos solo de CPU si es necesario
fi

mkdir -p "${JOB_DIR}/${output_dir_name}"

for m in $(seq $m_start $m_step $m_end); do

    SLURM_SCRIPT="${JOB_DIR}/job_MCMG_${m}.slurm"

    cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=${Nskip}_${lam}_${batch}_${Lattice}_${device}_${warm}_${meassurements}_${m}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/${output_dir_name}/specs_data_${m}.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/${output_dir_name}/specs_data_${m}.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
${extraline}
#SBATCH --cpus-per-task=${cpus_per_task_val}
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
echo "Running on CPUs: \$(scontrol show hostnames \$SLURM_NODELIST | tr '\n' ' ')"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "In directory: $(pwd)"


echo "Starting Python at $(date)"
python3 run_MCMG.py -Nskip $Nskip -lam $lam -batch $batch -L $Lattice -dev $device -warm $warm -meas $meassurements -m $m -mcmg $mcmg -train $train -Nlayers $Nlayers -width $width -Nlevels $Nlevels -snet $snet
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
