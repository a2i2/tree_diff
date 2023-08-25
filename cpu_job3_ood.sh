#!/bin/bash
#SBATCH --job-name="Merging Neural network 3"
#SBATCH --output=cpu_job3_ood.out
#SBATCH --error=cpu_job3_ood.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anupam.chaudhuri@deakin.edu.au

module purge
module load Anaconda3
source activate merge
python test_NN.py subset_0.7_higgs data_higgs_ood
