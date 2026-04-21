#!/bin/bash -l
#SBATCH --partition=angsd_class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=preprocess_ctrpv2
#SBATCH --time=48:00:00 # HH/MM/SS
#SBATCH --mem=40G # memory requested, units available : K,M,G,T
#SBATCH --cpus-per-task=16

source ~/.bashrc
conda activate sequential_prediction_311

python preprocess_ctrpv2.py \
    --ctrpv2_zip      ../../data/cell_viability_data/CTRPv2.0_2015_ctd2_ExpandedDataset.zip \
    --train_path      ../../data/perturbation_data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz \
    --inst_info       ../../data/perturbation_data/GSE92742_Broad_LINCS_inst_info.txt.gz \
    --drug_info_path  ../../data/perturbation_data/GSE92742_Broad_LINCS_pert_info.txt.gz \
    --landmark_genes  ../../data/perturbation_data/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz \
    --drug_format     maccs \
    --max_log10_conc_delta 0.2 \
    --processed_data_dir ../../data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl