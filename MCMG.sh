#!/bin/bash

# Usage check
if [ $# -ne 6 ]; then
    echo "Usage: $0 <kernelname> <initial> <final> <step>"
    echo "Example: $0 rbf_logrbf_ln= 1.0 3.0 0.5"
    exit 1
fi

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <KERNEL_NAME>"
    exit 1
fi

if [[ $(hostname) == fm* ]]; then
    EXCLUDE="fm01,fm04,fm08,fm24"
else
    EXCLUDE=""
fi

# Parameters
kernelname=$1   # e.g., rbf_logrbf_ln=
minit=$2      # e.g., -0.58
mfin=$3        # e.g., -0.54
step=$4         # e.g., 0.005
data=$5
modes=$6

# Loop over floating-point values using awk
value=$minit
while (( $(echo "$value <= $mfin" | bc -l) )); do
    # Format value to 2 decimal
    value_formatted=$(printf "%.2f" "$value")
    parameter="${kernelname}${value_formatted}"

    # Submit job #linh is the new grid
    sbatch run_IS.csh "PDF_N g_flat" "$modes" "log_lin" "$parameter" "$data"
    echo "Submitted: sbatch run_IS.csh \"PDF_N g_flat\" \"$modes\" \"log_lin\" \"$parameter\" \"$data\""

    # Increment
    value=$(echo "$value + $step" | bc -l)
done