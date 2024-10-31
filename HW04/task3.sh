#!/usr/bin/env bash

#SBATCH --job-name=nbody_experiment
#SBATCH --output=nbody_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

if [ "$1" = "compile" ]; then
  g++ task3.cpp -Wall -O3 -std=c++17 -fopenmp -o task3
elif [ "$1" = "run" ]; then
  particles=100
  end_time=100.0
  executable=task3
  for schedule in static dynamic guided; do
      for threads in {1..8}; do
          echo "Running with $threads threads and $schedule scheduling"
          ./$executable $particles $end_time $threads $schedule
      done
  done
fi
