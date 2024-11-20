#!/usr/bin/env bash
#SBATCH --job-name=task2_scaling
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --output="task2_scaling.out"
#SBATCH --error="task2_scaling.err"
#SBATCH --gres=gpu:1

# Load required modules
module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the code
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

# Output file for results
OUTPUT_FILE="task2_timing_results.csv"
echo "n,R,threads_per_block,time_ms" > $OUTPUT_FILE

# Parameters
R=128
THREADS_BLOCK1=1024
THREADS_BLOCK2=256

# Run for each value of n and both thread configurations
for n in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216; do
    for THREADS in $THREADS_BLOCK1 $THREADS_BLOCK2; do
        echo "Running task2 with n=$n, R=$R, threads_per_block=$THREADS..."
        
        # Run the program and extract timing information
        TIME=$(./task2 $n $R $THREADS | grep "Execution time" | awk '{print $3}')
        
        # Save the result
        echo "$n,$R,$THREADS,$TIME" >> $OUTPUT_FILE
    done
done
