#!/usr/bin/env bash

#SBATCH --cpus-per-task=20
#SBATCH -p instruction
#SBATCH -J task3_ts
#SBATCH -o task3_ts.out -e task3_ts.err

if [ "$1" = "compile" ]; then
  # Compile the task3.cpp and msort.cpp with OpenMP support
  g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

elif [ "$1" = "run" ]; then
  # Run task3 with n = 10^6, t = 8, and ts from 2^1 to 2^10
  n=1000000
  t=8
  echo "Running with n = $n and t = $t"
  for ts in 2 4 8 16 32 64 128 256 512 1024; do
    echo "Running with ts = $ts"
    ./task3 $n $t $ts >> task3_ts_times.txt  # Store the time output in a file
    echo
  done

elif [ "$1" = "plot" ]; then
  # Generate the plot using Python
  python task3_plot.py

elif [ "$1" = "clean" ]; then
  # Clean up generated files
  rm -f task3 task3.err task3_ts.out task3.pdf task3_ts_times.txt

else
  echo "./task3_ts.sh [compile | run | plot | clean]"
fi
