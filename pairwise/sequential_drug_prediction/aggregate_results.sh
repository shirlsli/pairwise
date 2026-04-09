#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train_eval_xpert_morgan_warm_start
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1

python -u aggregate_results.py --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_parallel --n_folds 5