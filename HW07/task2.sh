#!/usr/bin/env bash
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:20:00
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"
#SBATCH --gres=gpu:4
#SBATCH --mem=32G


module purge
module load nvidia/cuda/11.8.0

# Compile the program with optimizations
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Parameters
threads_per_block=1024

# Avoid invalid characters in loops and optimize resource usage
for exp in {10..30}; do
    n=$((2**exp))
    ./task1 $n $threads_per_block
done
