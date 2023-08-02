#!/bin/bash
#SBATCH --job-name="Vis Neural network"
#SBATCH --output=vis_job.out
#SBATCH --error=vis_job.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anupam.chaudhuri@deakin.edu.au

module purge
module load Anaconda3
source activate merge
python test_NN_vis.py
