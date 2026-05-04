#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train_xgboost
#SBATCH --mem=40G # memory requested, units available : K,M,G,T
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

DATA=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl
OUT=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression

# python train_ridge_regression.py \
#     --model_type gridsearch_xgboost \
#     --processed_data_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl \
#     --model_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression/xgboost_model.pkl \
#     --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression/xgboost_pcc_results.pkl \
#     --gs_folds 5 \
#     --n_jobs -1

# python train_ridge_regression.py \
#   --model_type xgboost_gene_only \
#   --processed_data_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl \
#   --model_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression/xgboost_model_gene_only.pkl \
#   --result_out /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/cell_viability_regression/xgboost_pcc_results_gene_only.pkl \
#   --gs_folds 5 \
#   --n_jobs -1

# python train_ridge_regression.py \
#   --model_type zero_shot_xgboost_gene_only \
#   --processed_data_path $DATA \
#   --model_out $OUT/xgb_gene_only_zero_shot_model.pkl \
#   --result_out $OUT/xgb_gene_only_zero_shot_results.pkl \
#   --n_estimators 300 \
#   --max_depth 6 \
#   --learning_rate 0.1 \
#   --subsample 0.7 \
#   --colsample_bytree 0.7 \
#   --test_frac 0.2 \
#   --zs_iters 20

# python train_ridge_regression.py \
#   --model_type gridsearch_xgboost_gene_only \
#   --processed_data_path $DATA \
#   --model_out $OUT/xgb_gene_only_gs_model.pkl \
#   --result_out $OUT/xgb_gene_only_gs_results.pkl \
#   --gs_folds 5 --n_jobs -1

python -u train_ridge_regression.py \
  --model_type zero_shot_xgboost \
  --processed_data_path $DATA \
  --model_out $OUT/xgb_zero_shot_model.pkl \
  --result_out $OUT/xgb_zero_shot_results.pkl \
  --n_estimators 300 \
  --max_depth 6 \
  --learning_rate 0.1 \
  --subsample 0.7 \
  --colsample_bytree 0.7 \
  --test_frac 0.2 \
  --zs_iters 20