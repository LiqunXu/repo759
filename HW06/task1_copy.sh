#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="task1.out"
#SBATCH --error="task1.err"

#SBATCH --gres=gpu:1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

./task1 1024 32