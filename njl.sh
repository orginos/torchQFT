#!/bin/sh

#SBATCH --job-name=NJL_setup
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchNJL/run_setup.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchNJL/run_setup.log


if [ $# -ne 11 ]; then
    echo "Usage: $0 <Nskip> <mass> <batch> <Lattice> <device> <warm> <meassurements> <g_start> <g_end> <g_step> <hmcsteps>"
    echo "Example: $0 1 0.0 50 16 cuda 100 300 0.1 1.0 0.1 14"
    exit 1
fi

if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm01,fm04,fm08,fm24"
else
    EXCLUDE=""
fi

Nskip=$1
m=$2
batch=$3
Lattice=$4
device=$5
warm=$6
meassurements=$7
g_start=$8
g_end=$9
g_step=${10} 
hmcsteps=${11}

output_dir_name="njldata_hmc"


JOB_DIR="/sciclone/pscr/yacahuanamedra/torchNJL"

# Check if device relates to GPU
if [[ "$device" == "cuda"* ]] || [ "$device" = "-1" ]; then
    extraline="#SBATCH --gres=gpu:1" # Solicita una GPU.
    cpus_per_task_val=4 # Núcleos de CPU para soportar la GPU.
else
    extraline=""
    cpus_per_task_val=8 # Más núcleos de CPU para ejecución solo en CPU.
fi

mkdir -p "${JOB_DIR}/${output_dir_name}"

for g in $(seq $g_start $g_step $g_end); do

    SLURM_SCRIPT="${JOB_DIR}/job_NJL_${g}.slurm"

    cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=NJL_${g}_${batch}_${Lattice}_${device}_${warm}_${meassurements}_${m}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchNJL/${output_dir_name}/specs_data_g${g}.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchNJL/${output_dir_name}/specs_data_g${g}.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
${extraline}
#SBATCH --cpus-per-task=${cpus_per_task_val}
#SBATCH --mem=2000M
#SBATCH --exclude=${EXCLUDE}

cd /sciclone/pscr/yacahuanamedra/torchNJL/


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
echo "In directory: \$(pwd)"


echo "Starting Python at \$(date)"
python3 run_njl.py --Nskip $Nskip --g $g --batch-size $batch --L $Lattice --device $device --Nwarm $warm --Nmeas $meassurements --mass $m --hmcsteps $hmcsteps
echo "Finished Python at \$(date)"
EOF
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "Job $JOB_ID submitted successfully."
            echo "Running with g = $g"
            rm -f "$SLURM_SCRIPT"

            echo "SLURM script $SLURM_SCRIPT deleted."
        else
            echo "Failed to submit job. SLURM script not deleted: $SLURM_SCRIPT"
        fi
done
