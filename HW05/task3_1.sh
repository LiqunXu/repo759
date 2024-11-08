#!/usr/bin/env bash
#SBATCH --job-name=task3
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:50:00
#SBATCH --output="task3.out"
#SBATCH --error="task3.err"

#SBATCH --gres=gpu:1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

# Compile the task3 CUDA program
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run the compiled program with a specified n (e.g., 1024)
./task3 1024
