#!/bin/sh

#SBATCH --job-name=MCMG_setup
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/ToDo/run1.log


if [ $# -ne 10 ]; then
    echo "Usage: $0 <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <mass>"
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
m=($8)
#models=($1)
#modes=($2)
#grids=($3)
#kernels=($4)
#data=($5)

# I do not need loops 
for kernel in ${kernels[@]}; do
  for model in ${models[@]}; do
    for mode in ${modes[@]}; do
      for grid in ${grids[@]}; do

        mkdir -p "${model}_${kernel}(${mode}+${grid})"
        SLURM_SCRIPT="${model}_${kernel}(${mode}+${grid})/job_script.slurm"
        

cat << EOF > $SLURM_SCRIPT
#!/bin/bash

#SBATCH --job-name=${Nskip}_${lam}_${batch}_${Lattice}_${device}_${warm}_${meassurements}_${m}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/${model}_${kernel}(${mode}+${grid})/specs_data_${m}.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/${model}_${kernel}(${mode}+${grid})/specs_data_${data}.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --exclude=${EXCLUDE}

cd /sciclone/pscr/yacahuanamedra/GP/

source ~/.bashrc
module load miniforge3/24.9.2-0
conda init
sleep 5
conda activate gptorch

echo "Python path: \$(which python3)"
python3 --version

echo "Running on node: \$(hostname)"
echo "Running on CPUs: \$(scontrol show hostnames \$SLURM_NODELIST)"
echo "SLURM job ID: \$SLURM_JOB_ID"
echo "SLURM job name: \$SLURM_JOB_NAME"
echo "In directory: $(pwd)"


echo "Starting Python at $(date)"
python3 run_IS.py --mean "$model" --ker "$kernel" --mode "$mode" --grid "$grid" --data "$data"
echo "Finished Python at $(date)"
EOF
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "Job $JOB_ID submitted successfully."
            echo "Running $model + $kernel in mode=$mode, grid=$grid"
            rm -f "$SLURM_SCRIPT"

            echo "SLURM script $SLURM_SCRIPT deleted."
        else
            echo "Failed to submit job. SLURM script not deleted: $SLURM_SCRIPT"
        fi

      done
    done
  done
done

