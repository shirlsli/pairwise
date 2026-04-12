#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=aggregate_results
#SBATCH --time=08:00:00 # HH/MM/SS
#SBATCH --mem=40G # memory requested, units available : K,M,G,T
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

python -u aggregate_results.py --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_parallel --n_folds 5