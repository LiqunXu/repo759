#!/usr/bin/env bash

#SBATCH --cpus-per-task=20
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o task3.out -e task3.err

# Optimal ts value based on previous experiments
optimal_ts=1024  # Replace with the value that yielded the best performance

if [ "$1" = "compile" ]; then
  # Compile the task3.cpp and msort.cpp with OpenMP support
  g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

elif [ "$1" = "run" ]; then
  # Run task3 with n = 10^6, t = 1 to 20, and the best ts value
  n=1000000
  echo "Running with n = $n and $optimal_ts"
  for t in {1..20}; do
    echo "Running with t = $t"
    ./task3 $n $t $optimal_ts 
    echo
  done

elif [ "$1" = "plot" ]; then
  # Generate the plot using Python
  python task3_plot.py

elif [ "$1" = "clean" ]; then
  # Clean up generated files
  rm -f task3 task3.err task3.out task3.pdf task3_threads_times.txt

else
  echo "./task3.sh [compile | run | plot | clean]"
fi
