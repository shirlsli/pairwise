import pandas as pd
import gzip
import json
import os
import shutil
import requests
import numpy as np
from cmapPy.pandasGEXpress import parse_gctx
from collections import Counter
from preprocess_drugs import preprocess_drugs
import pickle
from train_val_test_split import compute_cell_line_baselines

def get_top_durations(sig_info_path, matched_brd_ids, n=3):
    brd_set = set(matched_brd_ids)
    duration_counts = Counter()

    open_fn = gzip.open if sig_info_path.endswith('.gz') else open
    with open_fn(sig_info_path, 'rt', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col = {h.strip(): i for i, h in enumerate(header)}

        for line in f:
            fields = line.rstrip('\n').split('\t')
            pert_id = fields[col['pert_id']].strip()
            if pert_id not in brd_set:
                continue

            try:
                dose = float(fields[col['pert_dose']].strip())
                unit = fields[col['pert_dose_unit']].strip()
                duration = float(fields[col['pert_time']].strip())

                # Only collect durations for 9-11 uM samples
                if 9 <= dose <= 11 and unit in ('uM', 'µM'):
                    duration_counts[duration] += 1

            except (ValueError, KeyError):
                pass

    top_n = duration_counts.most_common(n)
    print(f"Top {n} most common durations for 9-11 µM exposures:")
    for duration, count in top_n:
        print(f"  {duration}h: {count} samples")

    return top_n

def get_brd_ids(drug_list_path):
    with open(drug_list_path, 'r', encoding='utf-8') as f:
        drugs = [line.strip() for line in f if line.strip()]

    drugs_lower = {d.lower(): d for d in drugs}

    params = {
        'filter': json.dumps({
            'where': {'pert_iname': {'inq': list(drugs_lower.keys())}},
            'fields': ['pert_id', 'pert_iname'],
        })
    }
    resp = requests.get(
        'https://api.clue.io/api/perts',
        params=params,
        headers={'user_key': os.getenv('CLUE_API_KEY', '')}, # export CLUE_API_KEY=your_api_key_here
        timeout=30,
    )
    resp.raise_for_status()

    matched = {}
    for record in resp.json():
        iname = record.get('pert_iname', '').lower()
        if iname in drugs_lower:
            original = drugs_lower[iname]
            if original not in matched:   # keep first hit
                matched[original] = record['pert_id']

    unmatched = [d for d in drugs if d not in matched]
    return matched, unmatched

def check_lincs_compounds(file_path, list_of_desired_compounds):
    matched, unmatched = get_brd_ids(list_of_desired_compounds)
    desired_set = set(matched.values())
    print(f"Desired compounds: {len(desired_set)}, Matched: {len(matched)}, Unmatched: {len(unmatched)}")

    open_fn = gzip.open if file_path.endswith('.gz') else open
    matches = set()
    with open_fn(file_path, 'rt', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if fields and fields[0].strip() in desired_set:
                matches.add(fields[0].strip())

    return matches

def filter_sig_info(sig_info_path, matched_brd_ids):
    brd_set = set(matched_brd_ids)
    exposure_24h = []
    dose_9_11uM = []

    open_fn = gzip.open if sig_info_path.endswith('.gz') else open
    with open_fn(sig_info_path, 'rt', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col = {h.strip(): i for i, h in enumerate(header)}

        for line in f:
            fields = line.rstrip('\n').split('\t')
            pert_id = fields[col['pert_id']].strip()
            if pert_id not in brd_set:
                continue

            # 24-hour exposure
            try:
                if float(fields[col['pert_time']].strip()) == 24:
                    exposure_24h.append(pert_id)
            except (ValueError, KeyError):
                pass
            # 9-11 µM dose
            try:
                dose = float(fields[col['pert_dose']].strip())
                unit = fields[col['pert_dose_unit']].strip()
                if 9 <= dose <= 11 and unit in ('uM', 'µM'):
                    dose_9_11uM.append(pert_id)
            except (ValueError, KeyError):
                pass

    return exposure_24h, dose_9_11uM

def explore_lincs():
    drug_list = '../../data/targeted_drug_list.txt'
    compound_file = '../../data/perturbation_data/compoundinfo_beta.txt'
    gse_file = '../../data/perturbation_data/GSE92742_Broad_LINCS_pert_info.txt.gz'
    sig_info_file = '../../data/perturbation_data/siginfo_beta.txt'
    sig_info_L1000_file = '../../data/perturbation_data/GSE92742_Broad_LINCS_sig_info.txt.gz'

    matches1 = check_lincs_compounds(compound_file, drug_list)
    matches2 = check_lincs_compounds(gse_file, drug_list)
    print("LINCS 2020 Total matches in compoundinfo_beta.txt: {}".format(len(matches1)))
    print("L1000 Total matches in GSE92742_LINCS_pert_info.txt.gz: {}".format(len(matches2)))

    exposure_24h, dose_9_11uM = filter_sig_info(sig_info_file, matches1)
    print("\nLINCS 2020 24-hour exposures: {} entries, {} unique".format(len(exposure_24h), len(set(exposure_24h))))
    print("\nLINCS 2020 9-11 µM dosages: {} entries, {} unique".format(len(dose_9_11uM), len(set(dose_9_11uM))))

    exposure_24h_L1000, dose_9_11uM_L1000 = filter_sig_info(sig_info_L1000_file, matches2)
    print("\nL1000 24-hour exposures: {} entries, {} unique".format(len(exposure_24h_L1000), len(set(exposure_24h_L1000))))
    print("\nL1000 9-11 µM dosages: {} entries, {} unique".format(len(dose_9_11uM_L1000), len(set(dose_9_11uM_L1000))))

    top_durations_L1000 = get_top_durations(sig_info_L1000_file, matches2)

def ensure_decompressed(gz_path):
    if not gz_path.endswith('.gz'):
        return gz_path
    out_path = gz_path[:-3]
    if not os.path.exists(out_path):
        print(f"Decompressing {gz_path} -> {out_path} (streaming)...")
        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)  # 64 MB chunks
        print("Done.")
    return out_path

def preprocess_data(args):
    # explore_lincs() # used to just manually looking at data
    print('Preprocessing LINCS data...')
    print('Extracting change in expression data...')
    gctx_path = ensure_decompressed(args.train_path) # '../../data/perturbation_data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz'
    print('Extracted change in expression data!')

    print('Extracting landmark genes...')
    landmark_genes = ensure_decompressed(args.landmark_genes) # '../../data/perturbation_data/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt.gz'
    print('Extracted landmark genes!')

    print('Filtering down to desired perturbations and controls...')
    inst_info = pd.read_table(ensure_decompressed(args.inst_info), sep='\t', header=0, low_memory=False)
    durations = args.dosage_durations
    drug_dosages = args.compound_concentrations
    inst_info_all = inst_info.copy()
    inst_info = inst_info[inst_info['pert_time'].isin(durations)]
    controls = inst_info[inst_info["pert_id"] == "DMSO"]
    perts = inst_info[
        (inst_info["pert_id"] != "DMSO") &
        (inst_info["pert_dose"].isin(drug_dosages))
    ]
    relevant_cell_lines = set(perts["cell_id"].unique())
    controls_all = inst_info_all[
        (inst_info_all["pert_id"] == "DMSO") &
        (inst_info_all["cell_id"].isin(relevant_cell_lines))
    ]

    perts_with_controls = []
    for (plate, time), perts_group in perts.groupby(["rna_plate", "pert_time"]):
        controls_group = controls[(controls["rna_plate"] == plate) & (controls["pert_time"] == time)]
        if not controls_group.empty:
            perts_group = perts_group.copy()
            control_ids = controls_group["inst_id"].tolist()
            perts_group["ctl_inst_ids"] = [control_ids] * len(perts_group)
            perts_with_controls.append(perts_group)
    perts = pd.concat(perts_with_controls).reset_index(drop=True)
    print('Finished filtering!')
    drug_format = args.drug_format

    print('Converting drug SMILES to {} fingerprints...'.format(drug_format))
    drugs_df = preprocess_drugs(format=drug_format,
        file_path=ensure_decompressed(args.drug_info_path), # '../../data/perturbation_data/GSE92742_Broad_LINCS_pert_info.txt.gz'
        pert_ids=perts['pert_id'].unique()
    )

    perts = perts[perts["pert_id"].isin(drugs_df["pert_id"])]
    print(f"Finished converting drug SMILES to {drug_format} fingerprints! {len(perts)} perturbations remaining.")
    return load_lincs_data(gctx_path, perts, drugs_df, landmark_genes, args, inst_info_all)

def load_lincs_data(gctx_path, filtered_perts, drugs_df, landmark_genes, args, inst_info_all):
    print('Decompressing gctz file if needed, might take awhile... so would recommend sbatch-ing this so you can go touch some grass while it runs :)')
    gctx_path = ensure_decompressed(gctx_path)
    print('Decompressed gctz file!')

    print('Parsing gctx file and extracting expression data for filtered perturbations and controls...')
    landmark_genes_col = pd.read_table(landmark_genes, sep='\t', header=0, low_memory=False)['pr_gene_id'].tolist()
    ctl_ids = filtered_perts['ctl_inst_ids'].tolist()
    ctl_ids_flat = [x for item in ctl_ids for x in (item if isinstance(item, list) else [item])]
    all_inst_ids = list(set(filtered_perts['inst_id'].tolist() + ctl_ids_flat))
    row_meta = parse_gctx.get_row_metadata(gctx_path)
    available_rids = set(row_meta.index.astype(str).tolist())
    landmark_genes_col = [g for g in [str(x) for x in landmark_genes_col] if g in available_rids]
    gctx_data = parse_gctx.parse(gctx_path, cid=all_inst_ids, rid=landmark_genes_col)
    expression_df = gctx_data.data_df
    print('Finished parsing gctx file and extracting expression data!')
    print('Computing cell line baseline profiles from DMSO controls...')
    cell_line_baselines = compute_cell_line_baselines(inst_info_all, expression_df, landmark_genes_col)
    print('Finished computing cell line baseline profiles!')
    drugs_df = drugs_df.set_index('pert_id')
    print('Constructing control-treatment pairs and corresponding drug fingerprints...')
    control_pairs = []
    treatment_pair = np.zeros((len(filtered_perts), len(landmark_genes_col)))
    drug_fp = np.zeros((len(filtered_perts), 167 if args.drug_format == 'maccs' else 1024))
    cell_ids, concentrations, durations, drug_names, rna_plates = [], [], [], [], []
    filtered_perts = filtered_perts.reset_index(drop=True)
    for i, row in filtered_perts.iterrows():
        if args.cell_line_consensus:
            control_pairs.append(cell_line_baselines[row['cell_id']].reshape(1, -1))
        else:
            ctl = row['ctl_inst_ids']
            control_pairs.append(expression_df[ctl].values.T.astype(np.float32))
        treatment_pair[i, :] = expression_df[row['inst_id']].values
        drug_fp[i, :] = drugs_df.loc[row['pert_id'], 'fp']
        cell_ids.append(row['cell_id'])
        concentrations.append(row['pert_dose'])
        durations.append(row['pert_time'])
        drug_names.append(row['pert_iname'])
        rna_plates.append(row['rna_plate'])

    save_path = args.processed_data_dir # '../../data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'control_pairs': control_pairs,
            'treatment_pair': treatment_pair,
            'drug_fp': drug_fp,
            'cell_ids': np.array(cell_ids),
            'concentrations': np.array(concentrations),
            'durations': np.array(durations),
            'drug_names': np.array(drug_names),
            'rna_plates': np.array(rna_plates),
            'cell_line_baselines': cell_line_baselines
        }, f)
    print('Finished constructing control-treatment pairs and corresponding drug fingerprints! Saved to {}'.format(save_path))
    return landmark_genes_col

def check_cell_types():
    with open('/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus_collapsed.pkl', 'rb') as f:
        data = pickle.load(f)
    cell_ids = data['cell_ids']
    num_cell_id = np.unique(cell_ids).size
    print(f"Number of unique cell lines: {num_cell_id}")
    print("Unique cell lines:")
    print(np.unique(cell_ids))