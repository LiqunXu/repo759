#!/usr/bin/env bash
#SBATCH --job-name=task12
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="task12.out"
#SBATCH --error="task12.err"
#SBATCH --gres=gpu:1

module purge
module load nvidia/cuda/11.8.0

# Compile the program with optimizations
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Parameters
n=384

# Avoid invalid characters in loops and optimize resource usage
for block in {4..12}; do
    ./task1 $n $block
done
