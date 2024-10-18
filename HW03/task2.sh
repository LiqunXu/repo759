#!/usr/bin/env bash

#SBATCH --cpus-per-task=20
#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err

if [ "$1" = "compile" ]; then
  # Compile the task2.cpp and matmul.cpp with OpenMP support
  g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

elif [ "$1" = "run" ]; then
  # Run task2 with n = 1024 and t = 1 to 20
  n=1024
  echo "Running with n = $n"
  for t in {1..20}; do
    echo "Running with t = $t"
    ./task2 $n $t >> task2_times.txt  # Store the time output in a file
    echo
  done

elif [ "$1" = "plot" ]; then
  # Generate the plot using Python
  python task2_plot.py

elif [ "$1" = "clean" ]; then
  # Clean up generated files
  rm -f task2 task2.err task2.out task2.pdf task2_times.txt

else
  echo "./task2.sh [compile | run | plot | clean]"
fi
