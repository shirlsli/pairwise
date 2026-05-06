#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=plot_viability_predictions
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

cd /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction

python -u plot_viability_predictions.py
