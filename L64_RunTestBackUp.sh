#!/bin/bash
#SBATCH -A NPH122
#SBATCH -p batch
#SBATCH -t 00:20:00
#SBATCH --nodes=3 -G 8 -c 56
#SBATCH --gpu-bind=none
#SBATCH -o l64_results_testbackup.out
###############SBATCH --array=0-1%1

rm -rf $HOME/.config/miopen
ln -s /tmp/miopen $HOME/.config/miopen


echo "
# w nl epochs
4 1 10
4 2 10
4 3 10
4 4 10
8 1 10
8 2 10
8 3 10
8 4 10
16 1 10
16 2 10
16 3 10
16 4 10
32 1 10
32 2 10
32 3 10
32 4 10
64 1 10
64 2 10
64 3 10
64 4 10" > epoch_table


srun -N 3 -n 3 -G 0 -l hostname | sort -n | awk '{print $2}' > nodes
machines=( `cat nodes` )

for((node=0 ; node<3 ; ++node )); do
        echo srun -r $node -N 1 -n 1 mkdir -p /tmp/miopen
        ssh ${machines[node]} mkdir -p /tmp/miopen &
done
wait

gpu=0
for g in 1.1; do
for w in 4 8 16 32 64; do
for nl in 1 2 3 4; do
        epochs="`grep "^$w $nl" epoch_table | awk '{print $3}'`"
        exp_dir="sm_phi4_L64_m-0.5_l${g}_w${w}_nl${nl}"
        if [ -d $exp_dir ]; then   
                extra_args="-f $exp_dir/sm_phi4_L64_m-0.5_l${g}_w${w}_nl${nl}.pt"
        else
                extra_args=""
        fi
        mkdir -p $exp_dir
	node="$(( gpu / 8 ))"
        echo ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" ssh ${machines[node]} python3 -m torch.distributed.run --nnodes=1 ...
	p="c-$g-$w-$nl.sh"
        cat << EOF > $p
. ~/env.sh
. ~/TorchQFT_v3/bin/activate
cd $PWD
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=NORMAL
ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" python3 -m torch.distributed.run --nnodes=1 \
                         --nproc_per_node=1 --master_port 2940$((gpu % 8)) train_stacked.py -d 3 -L 64 -b 200 -sb 40 \
                         -w $w -nl $nl -g $g -e $epochs -se 1 -lr 0.001 \
                         $extra_args &> $exp_dir/L64_W${w}_NL${nl}_G${g}_E${epoch}_LR0.001_B200_SB40_SE_100_D3
EOF
        ssh ${machines[node]} bash $PWD/$p &
        gpu="$((gpu+1))"

done
done
done
wait
