#!/bin/bash
#SBATCH -A NPH122
#SBATCH -p batch
#SBATCH -t 00:120:00
#SBATCH --nodes=15 -c 56
#SBATCH --gpu-bind=none
#SBATCH -o l32d3resultstestbackup.out
#SBATCH --array=0-56%1

rm -rf $HOME/.config/miopen
ln -s /tmp/miopen $HOME/.config/miopen

echo "
# w nl epochs
4 1 210
4 2 140
4 3 100
4 4 90
8 1 210
8 2 140
8 3 100
8 4 90
16 1 210
16 2 140
16 3 100
16 4 90
32 1 210
32 2 140
32 3 100
32 4 90
48 1 210
48 2 140
48 3 100
48 4 90
64 1 210
64 2 140
64 3 100
64 4 90"> epoch_table

srun -N 15 -n 15 -G 0 -l hostname | sort -n | awk '{print $2}' > nodes
machines=( `cat nodes` )

for((node=0 ; node<15 ; ++node )); do
        echo srun -r $node -N 1 -n 1 mkdir -p /tmp/miopen
        ssh ${machines[node]} mkdir -p /tmp/miopen &
done
wait

gpu=0
for g in 1.1 1.3 1.5 1.7 1.9; do
for w in 4 8 16 32 48 64; do
for nl in 1 2 3 4; do
        epochs="`grep "^$w $nl" epoch_table | awk '{print $3}'`"
        exp_dir="sm_phi4_L32_m-0.5_l${g}_w${w}_nl${nl}"
        if [ -d $exp_dir ]; then   
                extra_args="-f $exp_dir/sm_phi4_L32_m-0.5_l${g}_w${w}_nl${nl}.pt"
        else
                extra_args=""
        fi
        mkdir -p $exp_dir
	node="$(( gpu / 8 ))"
        echo ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" ssh ${machines[node]} python3 -m torch.distributed.run --nnodes=1 ...
	p="c-32-$g-$w-$nl.sh"
        cat << EOF > $p
. ~/env.sh
. ~/TorchQFT_v3/bin/activate
cd $PWD
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=NORMAL
ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" python3 -m torch.distributed.run --nnodes=1 \
                         --nproc_per_node=1 --master_port 2940$((gpu % 8)) train_stacked.py -d 3 -L 32 -b 200 -sb 40 \
                         -w $w -nl $nl -g $g -e $epochs -se 1 -lr 0.001 \
                         $extra_args &> $exp_dir/L32_W${w}_NL${nl}_G${g}_E${epoch}_LR0.001_B200_SB40_SE_100_D3
EOF
        ssh ${machines[node]} bash $PWD/$p &
        gpu="$((gpu+1))"

done
done
done
wait
