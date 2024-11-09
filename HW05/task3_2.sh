#!/usr/bin/env bash
#SBATCH --job-name=task3_batch
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-03:00:00   # Set sufficient time for all runs
#SBATCH --output="task3_batch.out"
#SBATCH --error="task3_batch.err"

#SBATCH --gres=gpu:1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the task3 CUDA program
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3_2

# Run task3 for each value of n and store the result
for ((i=10; i<=29; i++)); do
    n=$((2**i))
    echo "Running task3 with n=$n"
    ./task3_2 $n >> results.txt
done
