#!/bin/bash
#SBATCH -A hadron
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --gpus-per-task=4

source /global/homes/o/orginos/.bashrc
conda activate torch

#module load python
#module load pytorch

cd ~/torchQFT
srun python3 train_stacked_multi.py -w 64 -nl 4 -b 256 -nb 1 -g 1.9 -e 5000 -f sm_phi4_16_m-0.5_l1.9_w_64_l_4_st_1.dict
