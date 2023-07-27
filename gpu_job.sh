#!/bin/bash
#SBATCH --job-name="Training Neural network"
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anupam.chaudhuri@deakin.edu.au

module purge
module load Anaconda3
source activate tensorflow-gpu
python High_boson.py
