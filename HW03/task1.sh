#!/usr/bin/env bash

#SBATCH --cpus-per-task=20
#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

if [ "$1" = "compile" ]; then
  # Compile the task1.cpp and matmul.cpp with OpenMP support
  g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

elif [ "$1" = "run" ]; then
  # Run task1 with n = 1024 and t = 1 to 20
  n=1024
  echo "Running with n = $n"
  for t in {1..20}; do
    echo "Running with t = $t"
    ./task1 $n $t >> task1_times.txt  # Store the time output in a file
    echo
  done

elif [ "$1" = "plot" ]; then
  # Generate the plot using Python
  python task1_plot.py

elif [ "$1" = "clean" ]; then
  # Clean up generated files
  rm -f task1 task1.err task1.out task1.pdf task1_times.txt

else
  echo "./task1.sh [compile | run | plot | clean]"
fi
