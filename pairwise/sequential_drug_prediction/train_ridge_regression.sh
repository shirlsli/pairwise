#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train_ridge_regression
#SBATCH --mem=40G # memory requested, units available : K,M,G,T
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

DATA=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl
OUT=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression
MODEL_PATH=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna

# python train_ridge_regression.py \
#   --processed_data_path $DATA \
#   --model_out $OUT/ridge_model.pkl \
#   --result_out $OUT/ridge_pcc_results.pkl \
#   --alpha 1.0 \
#   --seed 42

# python train_ridge_regression.py \
#   --model_type zero_shot \
#   --processed_data_path $DATA \
#   --model_out $OUT/ridge_model_zero_shot_cell_line.pkl \
#   --result_out $OUT/ridge_pcc_results_zero_shot_cell_line.pkl \
#   --alpha 1.0 \
#   --split_by cell_line \
#   --test_frac 0.2 \
#   --zs_iters 20

# python train_ridge_regression.py \
#   --model_type zero_shot \
#   --processed_data_path $DATA \
#   --model_out $OUT/ridge_model_zero_shot_both.pkl \
#   --result_out $OUT/ridge_pcc_results_zero_shot_both.pkl \
#   --alpha 1.0 \
#   --split_by both \
#   --test_frac 0.2 \
#   --zs_iters 20

# python -u train_ridge_regression.py \
#   --model_type zero_shot_gene_only \
#   --processed_data_path $DATA \
#   --model_out $OUT/ridge_gene_only_model_zero_shot.pkl \
#   --result_out $OUT/ridge_gene_only_pcc_results_zero_shot.pkl \
#   --alpha 1.0 \
#   --test_frac 0.2 \
#   --zs_iters 20

# python train_ridge_regression.py \
#   --model_type zero_shot \
#   --processed_data_path $DATA \
#   --model_out $OUT/ridge_model_zero_shot_with_fp.pkl \
#   --result_out $OUT/ridge_pcc_results_zero_shot_with_fp.pkl \
#   --alpha 1.0 \
#   --test_frac 0.2 \
#   --zs_iters 20

# python train_ridge_regression.py \
#   --model_type ridge_synergy_inference \
#   --model_path $OUT/ridge_model.pkl \
#   --processed_data_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna/synergy_input.pkl \
#   --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna/inference_results.pkl

python train_ridge_regression.py \
  --model_type rank_synergy_predictions \
  --processed_data_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna/inference_results.pkl \
  --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/ranked_viability.csv \
  --lincs_data_path $DATA

# python train_ridge_regression.py \
#   --model_type ridge_synergy_inference \
#   --model_path $OUT/ridge_model.pkl \
#   --processed_data_path $MODEL_PATH/synergy_input.pkl \
#   --result_out $MODEL_PATH/inference_results.pkl \
#   --combine_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/targeted_drug_list_with_cell_lines.tsv
