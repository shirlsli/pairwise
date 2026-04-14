#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hyperparameter_search
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sequential_prediction_311

python -u pipeline.py \
    --config_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/configs/config_lincs_l1000.json \
    --cell_line_consensus \
    --drug_format morgan \
    --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna \
    --processed_data_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl \
    --patience 50 \
    --hyperparameter_search \
    --n_trials 20 \
    --warm_start \
    --fold_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/folds_warm_start \
    --batch_size 256 \
    --ppi_gene_vector_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/PPI_gene_vector_128d.npy