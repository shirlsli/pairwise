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

python train_ridge_regression.py \
  --model_type zero_shot \
  --processed_data_path $DATA \
  --model_out $OUT/ridge_model_zero_shot_with_fp.pkl \
  --result_out $OUT/ridge_pcc_results_zero_shot_with_fp.pkl \
  --alpha 1.0 \
  --test_frac 0.2 \
  --zs_iters 20
