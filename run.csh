#!/bin/bash

#SBATCH --output=/sciclone/pscr/yacahuanamedra/torchQFT/test/output.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/torchQFT/test/error.log


# Usage check gpus=-1 and cpu=0
if [ $# -ne 10 ]; then
    echo "Usage: $0 <Nskip> <lam> <batch> <Lattice> <device> <warm> <meassurements> <initial> <final> <step>"
    echo "Example: $0 1 2.4 10 8 -1 1000 10000 -0.58 -0.54 0.005"
    exit 1
fi


#working nodes in cpu only
if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm01,fm04,fm08,fm24"
else
    EXCLUDE=""
fi

# Parameters
Nskip=$1
lam=$2 
batch=$3
Lattice=$4 
device=$5
warm=$6
meassurements=$7
minit=$8
mfin=$9
step=$10

# Loop over floating-point values using awk
value=$minit
while (( $(echo "$value <= $mfin" | bc -l) )); do
    # Format value to 2 decimal
    value_formatted=$(printf "%.3f" "$value")
    parameter="${kernelname}${value_formatted}"

    # Submit job for several mass parameters
    #sbatch MCMG.sh "PDF_N g_flat" "$modes" "log_lin" "$parameter" "$data"
    echo "Submitted: sbatch MCMG.csh \" The field configuration was submitted with mass=$value_formatted\" "

    # Increment
    value=$(echo "$value + $step" | bc -l)
done