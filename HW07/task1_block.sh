#!/usr/bin/env bash
#SBATCH --job-name=task12
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="task12.out"
#SBATCH --error="task12.err"

#SBATCH --gres=gpu:1
module purge
module load nvidia/cuda/11.8.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Parameters
n=16384



for block in 1， 2， 3， 4， 5， 6， 7， 8，9， 10， 11， 12; do
    for X in $n; do
        ./task1 $n $block

    done
done