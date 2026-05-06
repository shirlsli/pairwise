#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train_eval_xpert_morgan_zero_shot_%a
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

source ~/.bashrc
conda activate sequential_prediction_311

python -u pipeline.py \
    --config_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/configs/config_optuna_best.json \
    --cell_line_consensus \
    --drug_format morgan \
    --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_zero_shot \
    --processed_data_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl \
    --zero_shot \
    --fold_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/folds_zero_shot \
    --ppi_gene_vector_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/PPI_gene_vector_977_gctx_order.npy \
    --fold_idx $SLURM_ARRAY_TASK_ID
