from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from geode import *
import numpy as np
from scipy.stats import spearmanr
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from xpert_morgan import compute_bin_edges, apply_binning

def make_split(idx, collapsed_data):
    return {
        'drug_fp':    collapsed_data['drug_fp'][idx],
        'ccl':        collapsed_data['control_pair'][idx],
        'ccl_binned': collapsed_data['control_pair_binned'][idx],
        'delta_expr': collapsed_data['delta_expr'][idx],
        'time_idx':   collapsed_data['time_idx'][idx],
        'cell_ids':   collapsed_data['cell_ids'][idx],
        'drug_names': collapsed_data['drug_names'][idx]
    }

def calculate_cd(control_matrix, treatment_matrix, landmark_gene_ids,
                 calculate_sig=1, nnull=100):
    rep_cds = []
    rep_pvals = []

    for i in range(treatment_matrix.shape[0]):
        treatment_rep = treatment_matrix[i:i+1]  # (1, 978)

        # geode expects (n_genes, n_samples)
        mat = np.hstack([
            control_matrix.T,   # (978, n_controls)
            treatment_rep.T     # (978, 1)
        ])
        col_labels = ['1'] * control_matrix.shape[0] + ['2']

        try:
            cd_result = chdir(
                mat, col_labels, landmark_gene_ids,
                calculate_sig=calculate_sig,
                nnull=nnull
            )

            gene_to_cd = dict(cd_result)
            cd_vector = np.array([
                gene_to_cd.get(str(g), 0.0)
                for g in landmark_gene_ids
            ], dtype=np.float32)

            rep_cds.append(cd_vector)

            if calculate_sig:
                rep_pvals.append(cd_result.get('pvalue', 1.0))

        except Exception as e:
            print(f"CD failed for replicate {i}: {e}")

    if len(rep_cds) == 0:
        return None, None

    # average across replicates then renormalize
    consensus_cd = np.mean(rep_cds, axis=0)
    norm = np.linalg.norm(consensus_cd)
    if norm > 0:
        consensus_cd = consensus_cd / norm

    mean_pval = np.mean(rep_pvals) if rep_pvals else None

    return consensus_cd, mean_pval

def calculate_modz(replicate_matrix):
    if replicate_matrix.shape[0] == 1:
        return replicate_matrix[0]
    
    if replicate_matrix.shape[0] == 2:
        return replicate_matrix.mean(axis=0)

    corr_matrix = spearmanr(replicate_matrix.T)[0]
    weights = np.sum(corr_matrix, axis=1) - 1  # exclude self-correlation
    weights = np.clip(weights, 0, None)

    if weights.sum() == 0:
        return replicate_matrix.mean(axis=0)

    weights = weights / weights.sum()
    return np.average(replicate_matrix, weights=weights, axis=0)

def compute_cell_line_baselines(inst_info, expression_df, landmark_genes_col):
    controls = inst_info[inst_info["pert_id"] == "DMSO"]
    cell_line_baselines = {}
    for cell_id, group in controls.groupby("cell_id"):
        # only use inst_ids that were actually loaded into expression_df
        valid_ids = [i for i in group["inst_id"].tolist() if i in expression_df.columns]
        if not valid_ids:
            continue
        dmso_profiles = expression_df[valid_ids].values.T.astype(np.float32)  # (n_dmso, 978)
        cell_line_baselines[cell_id] = calculate_modz(dmso_profiles)  # (978,)
    return cell_line_baselines

def collapse_replicates(data, args, landmark_gene_ids, calculate_sig=1, nnull=100, p_threshold=0.1):
    control_pairs = data['control_pairs']
    treatment_pairs = data['treatment_pair']
    drug_fp = data['drug_fp']
    drug_names = data['drug_names']
    cell_ids = data['cell_ids']
    durations = data['durations']
    rna_plates = data['rna_plates']
    df = pd.DataFrame({
        'drug_name': drug_names,
        'cell_id': cell_ids,
        'duration': durations,
        'rna_plate': rna_plates
    })

    group_cols = ['drug_name', 'cell_id', 'duration']

    results = []
    filtered = 0
    failed = 0

    for group_key, group_idx in df.groupby(group_cols).groups.items():
        idx = group_idx.values

        control_reps = control_pairs[idx[0]]
        if args.cell_line_consensus:
            modz_control = data['cell_line_baselines'][group_key[1]]
        else:
            control_reps = control_pairs[idx[0]]
            modz_control = calculate_modz(control_reps)
        treatment_reps = treatment_pairs[idx]

        modz_treatment = calculate_modz(treatment_reps)
        modz_delta = modz_treatment - modz_control

        if args.use_cd:
            cd_vector, p_val = calculate_cd(
                control_matrix=control_reps,
                treatment_matrix=treatment_reps,
                landmark_gene_ids=landmark_gene_ids,
                calculate_sig=calculate_sig,
                nnull=nnull
            )

            if cd_vector is None:
                # CD failed — fall back to MODZ delta
                print(f"CD failed for {group_key}, falling back to MODZ")
                failed += 1
                delta_expr = modz_delta
            else:
                # filter insignificant groups
                if calculate_sig and p_val is not None:
                    if p_val > p_threshold:
                        filtered += 1
                        continue

                # scale CD unit vector by MODZ magnitude
                magnitude = np.linalg.norm(modz_delta)
                delta_expr = cd_vector * magnitude
        else:
            delta_expr = modz_delta

        results.append({
            'delta_expr': delta_expr.astype(np.float32),
            'control_pair': modz_control.astype(np.float32),
            'drug_fp': drug_fp[idx[0]],
            'drug_name': group_key[0],
            'cell_id': group_key[1],
            'duration': group_key[2]
        })

    print(f"Before collapsing: {len(drug_names)} samples")
    print(f"After collapsing:  {len(results)} groups")
    if args.use_cd:
        print(f"Filtered by CD significance: {filtered}")
        print(f"CD failed, fell back to MODZ: {failed}")

    # return collapsed data — all arrays now have len(results) rows
    return {
        'delta_expr':   np.vstack([r['delta_expr'] for r in results]),
        'control_pair': np.vstack([r['control_pair'] for r in results]),
        'drug_fp':      np.vstack([r['drug_fp'] for r in results]),
        'drug_names':   np.array([r['drug_name'] for r in results]),
        'cell_ids':     np.array([r['cell_id'] for r in results]),
        'durations':    np.array([r['duration'] for r in results])
    }

def train_val_test_split(preprocessed_save_path, args, landmark_gene_ids):
    with open(preprocessed_save_path, 'rb') as f:
        data = pickle.load(f)

    normalized_data = collapse_replicates(data, args, landmark_gene_ids)

    # Compute time bin indices from raw durations, then drop raw durations
    normalized_data['time_idx'] = encode_time_bins(normalized_data['durations'])

    # Compute bin edges from a proxy training set (~90% of data).
    # Using all non-fold-0 drugs as proxy — edges are stable across
    # folds since MODZ scores have a consistent population distribution.
    drug_names = normalized_data['drug_names']
    unique_drugs = np.unique(drug_names)
    np.random.seed(42)
    shuffled_drugs = np.random.permutation(unique_drugs)
    folds_preview = [f.tolist() for f in np.array_split(shuffled_drugs, 10)]
    proxy_train_mask = ~np.isin(drug_names, folds_preview[0])
    proxy_train_control = normalized_data['control_pair'][proxy_train_mask]

    bin_edges = compute_bin_edges(proxy_train_control, n_bins=64)
    normalized_data['control_pair_binned'] = apply_binning(
        normalized_data['control_pair'], bin_edges
    )

    collapsed_save_path = preprocessed_save_path.replace('.pkl', '_collapsed.pkl')
    with open(collapsed_save_path, 'wb') as f:
        pickle.dump({
            'delta_expr':          normalized_data['delta_expr'],
            'control_pair':        normalized_data['control_pair'],
            'control_pair_binned': normalized_data['control_pair_binned'],
            'bin_edges':           bin_edges,
            'drug_fp':             normalized_data['drug_fp'],
            'drug_names':          normalized_data['drug_names'],
            'cell_ids':            normalized_data['cell_ids'],
            'time_idx':            normalized_data['time_idx'],
            'cell_line_baselines': data.get('cell_line_baselines', {}),
        }, f)

    splits = create_kfold_splits(normalized_data, args, landmark_gene_ids,
                                  n_folds=10, random_seed=42)
    return splits, collapsed_save_path

TIME_BIN_EDGES = np.array([0, 3, 6, 12, 24, 48, 72])

def encode_time_bins(durations, bin_edges=TIME_BIN_EDGES):
    """Map continuous durations to discrete bin indices."""
    return (np.digitize(durations, bin_edges[1:-1])
              .astype(np.int64))

def create_kfold_splits(collapsed_data, args,
                         landmark_gene_ids, n_folds=10,
                         random_seed=42):
    drug_names = collapsed_data['drug_names']
    unique_drugs = np.unique(drug_names)

    np.random.seed(random_seed)
    shuffled_drugs = np.random.permutation(unique_drugs)

    folds = [fold.tolist() for fold in np.array_split(shuffled_drugs, n_folds)]
    print(f"\nCreated {len(folds)} folds:")
    for i, fold in enumerate(folds):
        fold_idx = np.where(np.isin(drug_names, fold))[0]
        print(f"  Fold {i}: {len(fold)} compounds, {len(fold_idx)} samples")

    all_splits = []
    np.random.seed(random_seed)

    for test_fold_idx in range(n_folds):
        test_drugs = set(folds[test_fold_idx])

        train_drugs = set()
        for i, fold in enumerate(folds):
            if i != test_fold_idx:
                train_drugs.update(fold)

        train_drug_list = np.array(list(train_drugs))
        np.random.shuffle(train_drug_list)
        n_val_compounds = max(1, len(train_drug_list) // n_folds)

        val_drugs = set(train_drug_list[:n_val_compounds])
        train_drugs = train_drugs - val_drugs

        train_idx = np.where(np.isin(drug_names, list(train_drugs)))[0]
        val_idx = np.where(np.isin(drug_names, list(val_drugs)))[0]
        test_idx = np.where(np.isin(drug_names, list(test_drugs)))[0]

        print(f"\nFold {test_fold_idx} as test:")
        print(f"  Train: {len(train_idx)} samples ({len(train_drugs)} compounds)")
        print(f"  Val:   {len(val_idx)} samples ({len(val_drugs)} compounds)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_drugs)} compounds)")

        all_splits.append({
            'fold':  test_fold_idx,
            'train': make_split(train_idx, collapsed_data),
            'val':   make_split(val_idx, collapsed_data),
            'test':  make_split(test_idx, collapsed_data)
        })
    return all_splits