#!/bin/bash -l
#SBATCH --partition=scu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=inference_sequential_drug_prediction
#SBATCH --mem-per-gpu=60GB
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sequential_prediction_311
python -u pipeline.py --inference --inference_path /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/best_models_parallel --drug_list /athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/targeted_drug_list.txt