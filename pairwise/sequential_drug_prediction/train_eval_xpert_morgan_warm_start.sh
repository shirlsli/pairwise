#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=train_eval_xpert_morgan_warm_start_fold_%a
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

source ~/.bashrc
conda activate sequential_prediction_311

python -u pipeline.py \
    --config_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/configs/config_lincs_l1000.json \
    --cell_line_consensus \
    --drug_format morgan \
    --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_parallel \
    --processed_data_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl \
    --patience 20 \
    --warm_start \
    --fold_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/folds_warm_start \
    --ppi_gene_vector_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/PPI_gene_vector_128d.npy \
    --batch_size 256 \
    --epochs 200 \
    --learning_rate 5.6e-3 \
    --fold_idx $SLURM_ARRAY_TASK_ID