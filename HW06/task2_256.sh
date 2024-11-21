#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"

#SBATCH --gres=gpu:1
module purge
module load nvidia/cuda/11.8.0


nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

# Parameters
THREADS_BLOCK1=256
R=128

# Run for each value of n and both thread configurations
for ((i=10; i<=29; i++)); do
    n=$((2**i))
    for THREADS in $THREADS_BLOCK1; do
        ./task2 $n $R $THREADS

    done
done

