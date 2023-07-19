#!/bin/bash
#SBATCH --job-name="Merging Neural network"
#SBATCH --output=cpu_job.out
#SBATCH --error=cpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module purge
module load Anaconda3
source activate merge
python test_NN.py
