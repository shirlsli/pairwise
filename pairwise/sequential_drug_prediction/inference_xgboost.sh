#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=inference_xgboost
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

cd /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction

DATA=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl
OUT=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression
SYNERGY_INPUT=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna/synergy_input.pkl
COMBINE_PATH=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/targeted_drug_list_with_cell_lines.tsv

# python -u train_ridge_regression.py \
#   --model_type xgboost_inference \
#   --model_path $OUT/xgboost_model.pkl \
#   --processed_data_path $DATA \
#   --result_out $OUT/xgboost_inference_results.pkl

# python -u train_ridge_regression.py \
#   --model_type xgboost_gene_only_inference \
#   --model_path $OUT/xgboost_model_gene_only.pkl \
#   --processed_data_path $DATA \
#   --result_out $OUT/xgboost_gene_only_inference_results.pkl

# python -u train_ridge_regression.py \
#   --model_type xgboost_synergy_inference \
#   --model_path $OUT/xgboost_model.pkl \
#   --processed_data_path $SYNERGY_INPUT \
#   --combine_path $COMBINE_PATH \
#   --result_out $OUT/xgboost_synergy_inference_results.pkl

# python -u train_ridge_regression.py \
#   --model_type rank_synergy_predictions \
#   --processed_data_path $OUT/xgboost_synergy_inference_results.pkl \
#   --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/ranked_xgboost_viability.csv

python -u train_ridge_regression.py \
  --model_type xgboost_gene_only_synergy_inference \
  --model_path $OUT/xgboost_model_gene_only.pkl \
  --processed_data_path $SYNERGY_INPUT \
  --combine_path $COMBINE_PATH \
  --result_out $OUT/xgboost_gene_only_synergy_inference_results.pkl

python -u train_ridge_regression.py \
  --model_type rank_synergy_predictions \
  --processed_data_path $OUT/xgboost_gene_only_synergy_inference_results.pkl \
  --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/ranked_xgboost_gene_only_viability.csv
