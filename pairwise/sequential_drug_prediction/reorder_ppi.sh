#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=reorder_ppi
#SBATCH --mem=40G # memory requested, units available : K,M,G,T
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

python -u reorder_ppi.py