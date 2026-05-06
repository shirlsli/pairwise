import pickle
import re
import pandas as pd
import numpy as np

PERT_INFO_PATH = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/GSE92742_Broad_LINCS_pert_info.txt'

def normalize_cell_name(name):
    name = re.sub(r'[\s\-]', '', name).lower()
    return name

def build_smiles_lookup(lincs_pkl_path, lincs_data=None, pert_info_path=PERT_INFO_PATH):
    if lincs_data is None:
        with open(lincs_pkl_path, 'rb') as f:
            lincs_data = pickle.load(f)
    lincs_inames = set(str(n).lower() for n in lincs_data['drug_names'])

    pert_info = pd.read_csv(pert_info_path, sep='\t', low_memory=False, usecols=['pert_iname', 'canonical_smiles'])
    pert_info = pert_info[pert_info['pert_iname'].str.lower().isin(lincs_inames)]
    pert_info = pert_info.dropna(subset=['canonical_smiles'])
    pert_info = pert_info[~pert_info['canonical_smiles'].isin(['-666', 'restricted'])]
    return dict(zip(pert_info['pert_iname'].str.lower(), pert_info['canonical_smiles']))

def save_cell_line_lists(p13_cell_lines, lincs_cell_ids, output_dir):
    p13_path = f"{output_dir}/p13_cell_lines.txt"
    lincs_path = f"{output_dir}/lincs_cell_lines.txt"
    with open(p13_path, 'w') as f:
        f.write('\n'.join(sorted(p13_cell_lines)))
    with open(lincs_path, 'w') as f:
        f.write('\n'.join(sorted(lincs_cell_ids)))
    print(f"Saved {len(p13_cell_lines)} p13 cell lines to {p13_path}")
    print(f"Saved {len(lincs_cell_ids)} LINCS cell lines to {lincs_path}")

def preprocess_p13(input_path, output_path, lincs_pkl_path, targeted_drugs_path):
    with open(lincs_pkl_path, 'rb') as f:
        lincs_data = pickle.load(f)
    print(f"lincs_data keys: {lincs_data.keys()}")
    print(f"Sample cell_ids: {list(lincs_data['cell_ids'])[:10]}")
    print(f"Type of cell_ids: {type(lincs_data['cell_ids'])}")
    lincs_cell_ids = set(lincs_data['cell_ids'])

    p13_df = pd.read_csv(input_path, low_memory=False)
    p13_df = p13_df[['drug_name_x', 'drug_name_y', 'cell_line_name', 'DepMap_ID',
                     'tissue_name', 'synergy_bliss', 'ri_row', 'ri_col']]
    p13_df.dropna(inplace=True)

    p13_cell_lines = set(p13_df['cell_line_name'].unique())
    save_cell_line_lists(p13_cell_lines, lincs_cell_ids, output_dir='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/synergy_data/p13')

    p13_norm = {normalize_cell_name(c): c for c in p13_cell_lines}
    lincs_norm = {normalize_cell_name(c): c for c in lincs_cell_ids}
    overlap = set(p13_norm.keys()) & set(lincs_norm.keys())
    print(f"P13 unique cell lines:   {len(p13_norm)}")
    print(f"LINCS cell lines:        {len(lincs_norm)}")
    print(f"Overlap after normalize: {len(overlap)}")

    before = len(p13_df)
    p13_df = p13_df[p13_df['cell_line_name'].map(normalize_cell_name).isin(overlap)]
    print(f"Dropped {before - len(p13_df)} rows with cell lines not in LINCS")

    smiles_lookup = build_smiles_lookup(lincs_pkl_path, lincs_data)
    print(f"Sample p13 drug names:   {p13_df['drug_name_x'].unique()[:5].tolist()}")
    print(f"Sample LINCS drug names: {list(smiles_lookup.keys())[:5]}")
    p13_df = add_smiles_columns(p13_df, smiles_lookup)
    p13_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    training_samples = subset_training(p13_df, targeted_drugs_path)
    training_output_path = output_path.replace('.csv', '_training.csv')
    training_samples.to_csv(training_output_path, index=False)
    print(f"Saved training samples to {training_output_path}")
    return p13_df, training_samples

def add_smiles_columns(p13_df, smiles_lookup):
    p13_df['smiles1'] = p13_df['drug_name_x'].str.lower().map(smiles_lookup)
    p13_df['smiles2'] = p13_df['drug_name_y'].str.lower().map(smiles_lookup)
    before = len(p13_df)
    p13_df = p13_df.dropna(subset=['smiles1', 'smiles2'])
    after = len(p13_df)
    print(f"Dropped {before - after} rows (drugs not in LINCS training set)")
    print(f"Final dataset size: {after}")
    return p13_df

def subset_training(p13_df, targeted_drugs_path):
    targeted = pd.read_csv(targeted_drugs_path, sep='\t', header=None, names=['cell_line', 'drug_name', 'count'])
    targeted_drugs = set(targeted['drug_name'].str.lower())

    drug_x = p13_df['drug_name_x'].str.lower()
    drug_y = p13_df['drug_name_y'].str.lower()

    targeted_mask = drug_x.isin(targeted_drugs) | drug_y.isin(targeted_drugs)

    training_samples = p13_df[~targeted_mask].copy()
    print(f"subset_training: {len(training_samples)} / {len(p13_df)} rows kept (removed {targeted_mask.sum()} targeted rows)")
    return training_samples

if __name__ == "__main__":
    input_path = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/synergy_data/p13/p13_trueset.csv'
    output_path = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/synergy_data/p13/p13_preprocessed.csv'
    processed_data_path = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl'
    targeted_drugs_path = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/targeted_drug_list.txt'
    preprocess_p13(input_path, output_path, processed_data_path)
