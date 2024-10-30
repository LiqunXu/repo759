#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err
#SBATCH --cpus-per-task=8

if [ "$1" = "compile" ]; then
  g++ task2.cpp -Wall -O3 -std=c++17 -o task2
elif [ "$1" = "run" ]; then
  echo "Running with number of particles simulation end time"
  ./task2 100 10
elif [ "$1" = "clean" ]; then
  rm -f task2 task2.err task2.out task2.pdf
else
  echo "./task2.sh [compile | run | clean]"
fi
