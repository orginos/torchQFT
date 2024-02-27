#!/bin/bash
#SBATCH -A NPH122
#SBATCH -p batch
#SBATCH -t 00:20:00
#SBATCH --nodes=3 -G 8 -c 56
#SBATCH --gpu-bind=none
#SBATCH -o l64_results_test_2.out
###############SBATCH --array=0-1%1

rm -rf $HOME/.config/miopen
ln -s /tmp/miopen $HOME/.config/miopen

export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=NORMAL

. ~/env.sh
. ~/TorchQFT_v3/bin/activate
#while sleep 60 ; do rocm-smi; done &

echo "
# w nl epochs
4 1 1
4 2 1
4 3 1
4 4 1
8 1 1
8 2 1
8 3 1
8 4 1
16 1 1
16 2 1
16 3 1
16 4 1
32 1 1
32 2 1
32 3 1
32 4 1
64 1 1
64 2 1
64 3 1
64 4 1" > epoch_table

#nodes=3
for((node=0 ; node<3 ; ++node )); do
        echo srun -r $node -N 1 -n 1 mkdir -p /tmp/miopen
        srun -r $node -N 1 -n 1 -G 0 -c 1 --overlap -s mkdir -p /tmp/miopen &
done
wait

gpu=0
for g in 1.1; do
for w in 4 8 16 32 64; do
for nl in 1 2 3 4; do
        epochs="`grep "^$w $nl" epoch_table | awk '{print $3}'`"
        exp_dir="sm_phi4_L64_m-0.5_l${g}_w${w}_nl${nl}"
        if [ -d $exp_dir ]; then
                last_epoch="`cat $exp_dir/last_epoch`"
                extra_args="-f $exp_dir/sm_phi4_L64_m-0.5_l${g}_w${w}_nl${nl}.pt"
        else
                last_epoch=0
                extra_args=""
        fi
        mkdir -p $exp_dir
        echo ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" srun -r "$(( gpu / 8 ))" -N 1 -n 1 -G 0 --overlap -s python3 -m torch.distributed.run --nnodes=1 ...
        ROCM_VISIBLE_DEVICES="$(( gpu % 8 ))" srun -r "$(( gpu / 8 ))" -N 1 -n 1 --overlap python3 -m torch.distributed.run --nnodes=1 \
                         --nproc_per_node=1 --master_port "2940$((gpu % 8))" train_stacked.py -d 3 -L 64 -b 200 -sb 40 \
                         -w $w -nl $nl -g $g -e $epochs -le $last_epoch -se 1 -lr 0.001 \
                         $extra_args &> $exp_dir/L64_W${w}_NL${nl}_G${g}_E${epoch}_LR0.001_B200_SB40_SE_100_D3 && \
                echo "$(( last_epoch + epochs ))" > $exp_dir/last_epoch &
        gpu="$((gpu+1))"

done
done
done
wait
