#!/usr/bin/env bash
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:30:00
#SBATCH --output="task2.out"
#SBATCH --error="task2.err"

#SBATCH --gres=gpu:1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the task2 CUDA program
nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run the compiled program
./task2
