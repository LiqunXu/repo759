#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"

#SBATCH --gres=gpu:1
module purge
module load nvidia/cuda/11.8.0


nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Parameters
THREADS_BLOCK1=256

# Run for each value of n and both thread configurations
for n in 32 64 128 256 512 1024 2048 4096 8192 16384; do
    for THREADS in $THREADS_BLOCK1; do
        ./task1 $n $THREADS

    done
done
