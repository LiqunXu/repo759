#!/usr/bin/env bash
#SBATCH --job-name=task1_scaling
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH --output="task1_scaling.out"
#SBATCH --error="task1_scaling.err"
#SBATCH --gres=gpu:1

# Load required modules
module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the code
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Output file for results
OUTPUT_FILE="timing_results.csv"
echo "n,threads_per_block,time_ms" > $OUTPUT_FILE

# Parameters
THREADS_BLOCK1=1024
THREADS_BLOCK2=256 # Second configuration for threads per block

# Run for each value of n and both thread configurations
for n in 32 64 128 256 512 1024 2048 4096 8192 16384; do
    for THREADS in $THREADS_BLOCK1 $THREADS_BLOCK2; do
        echo "Running task1 with n=$n and threads_per_block=$THREADS..."
        
        # Run the program and extract timing information
        TIME=$(./task1 $n $THREADS | grep "Execution time" | awk '{print $3}')
        
        # Save the result
        echo "$n,$THREADS,$TIME" >> $OUTPUT_FILE
    done
done
