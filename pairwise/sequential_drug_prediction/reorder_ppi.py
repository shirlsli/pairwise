import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress import parse_gctx

DATA = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data'

ppi = np.load(f'{DATA}/perturbation_data/PPI_gene_vector_128d.npy', allow_pickle=True)

gene_info = pd.read_table(f'{DATA}/perturbation_data/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt',
                          sep='\t', header=0, low_memory=False)
gene_info_order = [str(x) for x in gene_info['pr_gene_id'].tolist()]
gene_info_row = {g: i for i, g in enumerate(gene_info_order)}

gctx_path = f'{DATA}/perturbation_data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx'
row_meta = parse_gctx.get_row_metadata(gctx_path)
gctx_all_order = row_meta.index.astype(str).tolist()

gene_info_set = set(gene_info_order)
gctx_landmark_order = [g for g in gctx_all_order if g in gene_info_set]

new_ppi = np.array([ppi[gene_info_row[g]] for g in gctx_landmark_order], dtype=np.float64)
print(f"new_ppi shape: {new_ppi.shape}")

out = f'{DATA}/perturbation_data/PPI_gene_vector_977_gctx_order.npy'
np.save(out, new_ppi)
print(f"Saved to {out}")
