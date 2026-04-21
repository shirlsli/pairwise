#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=optuna_trials
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-17

# Each array task runs 1 Optuna trial and writes to a shared JournalFile.
# 2 trials already completed (slurm-2788447) are seeded beforehand, so only
# 18 new workers are needed to reach 20 total trials.
#
# Workflow:
#   python pipeline.py ... --seed_optuna --optuna_storage $OPTUNA_STORAGE
#   JOBID=$(sbatch --parsable hyperparameter_search_parallel.sh)
#   sbatch --dependency=afterok:$JOBID hyperparameter_search_collect.sh

source ~/.bashrc
conda activate sequential_prediction_311

OPTUNA_STORAGE=/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/optuna_journal.log

python -u pipeline.py \
    --config_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/configs/config_lincs_l1000.json \
    --cell_line_consensus \
    --drug_format morgan \
    --model_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_optuna \
    --processed_data_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl \
    --patience 20 \
    --hyperparameter_search \
    --n_trials 18 \
    --n_trials_per_worker 1 \
    --optuna_storage "$OPTUNA_STORAGE" \
    --search_only \
    --warm_start \
    --fold_output_dir /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/folds_warm_start \
    --batch_size 256 \
    --ppi_gene_vector_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/PPI_gene_vector_128d.npy
