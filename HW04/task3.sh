#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o task3.out -e task3.err
#SBATCH --cpus-per-task=8

if [ "$1" = "compile" ]; then
  g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
elif [ "$1" = "run" ]; then
  echo "Running with number of particles, simulation end time, num thread" >> task3.out
  ./task3 100 100 8 >> task3.out 2>&1
elif [ "$1" = "clean" ]; then
  rm -f task3 task3.err task3.out task3.pdf
else
  echo "./task3.sh [compile | run | clean]"
fi
