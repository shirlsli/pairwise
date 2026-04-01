#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=model_comparison
#SBATCH --mem=80G # memory requested, units available : K,M,G,T
#SBATCH --gres=gpu:a40:1

source ~/.bashrc
conda activate sequential_prediction_311

python pipeline.py \
    --cell_line_consensus \
    --drug_format morgan \
    --hyperparameter_search \
    --n_trials 20 \
    --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models \
    --processed_data_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl