import pandas as pd
import numpy as np
import torch
import gzip
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from preprocess_lincs import preprocess_data, ensure_decompressed
from train_val_test_split import train_val_test_split
import argparse
import pickle
from torch.utils.data import DataLoader
from cmapPy.pandasGEXpress import parse_gctx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from xpert_morgan import XPertMorgan, PerturbationDataset, evaluate_model
from hyperparameter_search import run_hyperparameter_search

def train_fold(split, args, device):
    fold_idx = split['fold']
    print(f"\n{'='*50}")
    print(f"Training fold {fold_idx}")
    print(f"{'='*50}")

    train_data = split['train']
    val_data   = split['val']
    test_data  = split['test']

    train_dataset = PerturbationDataset(
        fingerprints=train_data['drug_fp'],
        binned_expr=train_data['ccl_binned'],
        time_idx=train_data['time_idx'],
        y=train_data['delta_expr']
    )
    val_dataset = PerturbationDataset(
        fingerprints=val_data['drug_fp'],
        binned_expr=val_data['ccl_binned'],
        time_idx=val_data['time_idx'],
        y=val_data['delta_expr']
    )
    test_dataset = PerturbationDataset(
        fingerprints=test_data['drug_fp'],
        binned_expr=test_data['ccl_binned'],
        time_idx=test_data['time_idx'],
        y=test_data['delta_expr']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = XPertMorgan(
        gene_number=train_data['delta_expr'].shape[1],
        fingerprint_dim=train_data['drug_fp'].shape[1],
        hidden_size=args.hidden_size,
        n_bins=args.n_bins,
        num_heads=args.num_heads,
        ctl_structure=args.ctl_structure,
        trt_structure=args.trt_structure,
        num_time_bins=6,
        mse_weight=args.mse_weight,
        pcc_weight=args.pcc_weight,
        learning_rate=args.learning_rate,
        epoch=args.epochs,
        device=device,
        drug_hidden_dim=args.drug_hidden_dim,
        model_file=os.path.join(
            args.model_output_dir,
            f"model_fold_{fold_idx}.pt"
        )
    )

    print(f"Training on {len(train_dataset)} samples...")
    model.fit(train_loader, valid_loader)

    # Load best checkpoint saved during training
    model = torch.load(
        os.path.join(args.model_output_dir, f"model_fold_{fold_idx}.pt"),
        map_location=device
    )

    print(f"Evaluating on test set with {len(test_dataset)} samples...")
    pccs = evaluate_model(model, test_loader)
    test_pcc = float(np.mean(pccs))
    test_pcc_std = float(np.std(pccs))
    print(f"Fold {fold_idx} test PCC: {test_pcc:.4f} ± {test_pcc_std:.4f}")

    return {
        'fold':         fold_idx,
        'test_pcc':     test_pcc,
        'test_pcc_std': test_pcc_std
    }


def get_landmark_gene_ids(landmark_genes_file, gctx_path):
    landmark_genes = ensure_decompressed(landmark_genes_file)
    landmark_genes_col = pd.read_table(
        landmark_genes, sep='\t', header=0, low_memory=False
    )['pr_gene_id'].tolist()
    gctx_path = ensure_decompressed(gctx_path)
    row_meta = parse_gctx.get_row_metadata(gctx_path)
    available_rids = set(row_meta.index.astype(str).tolist())
    landmark_genes_col = [
        g for g in [str(x) for x in landmark_genes_col]
        if g in available_rids
    ]
    return landmark_genes_col


def seq_pred_pipeline(args):
    device = args.device
    os.makedirs(args.model_output_dir, exist_ok=True)

    if not os.path.exists(args.processed_data_dir):
        landmark_gene_ids = preprocess_data(args)
    else:
        landmark_gene_ids = get_landmark_gene_ids(
            args.landmark_genes, args.train_path
        )

    all_splits, collapsed_path = train_val_test_split(
        args.processed_data_dir, args, landmark_gene_ids
    )

    os.makedirs(args.fold_output_dir, exist_ok=True)
    for split in all_splits:
        fold_path = os.path.join(
            args.fold_output_dir, f"fold_{split['fold']}.pkl"
        )
        if not os.path.exists(fold_path):
            with open(fold_path, 'wb') as f:
                pickle.dump(split, f)
            print(f"Saved fold {split['fold']} to {fold_path}")

    if args.hyperparameter_search:
        print("Running hyperparameter search...")
        best_params = run_hyperparameter_search(
            all_splits, args.device,
            n_trials=args.n_trials
        )
        # Override args with best params
        args.hidden_size     = best_params['hidden_size']
        args.num_heads       = best_params['num_heads']
        args.ctl_structure   = best_params['ctl_structure']
        args.trt_structure   = best_params['trt_structure']
        args.learning_rate   = best_params['learning_rate']
        args.mse_weight      = best_params['mse_weight']
        args.pcc_weight      = best_params['pcc_weight']
        args.drug_hidden_dim = best_params['drug_hidden_dim']
        print(f"Best params found: {best_params}")
    fold_results = []
    for split in all_splits:
        result = train_fold(split, args, device)
        fold_results.append(result)

    pccs = [r['test_pcc'] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"Cross-validation results ({len(pccs)} folds):")
    print(f"Mean PCC: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}")
    print(f"Min PCC:  {np.min(pccs):.4f}")
    print(f"Max PCC:  {np.max(pccs):.4f}")
    print(f"{'='*50}")

    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential Drug Prediction Pipeline"
    )
    # Data paths
    parser.add_argument('--train_path', type=str,
        default='../../data/perturbation_data/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
    parser.add_argument('--landmark_genes', type=str,
        default='../../data/perturbation_data/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt')
    parser.add_argument('--inst_info', type=str,
        default='../../data/perturbation_data/GSE92742_Broad_LINCS_inst_info.txt.gz')
    parser.add_argument('--drug_info_path', type=str,
        default='../../data/perturbation_data/GSE92742_Broad_LINCS_pert_info.txt.gz')
    parser.add_argument('--processed_data_dir', type=str,
        default='../../data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM.pkl')
    parser.add_argument('--fold_output_dir', type=str,
        default='../../data/perturbation_data/folds')
    parser.add_argument('--model_output_dir', type=str,
        default='../../data/models')

    # Preprocessing
    parser.add_argument('--drug_format', type=str,
        default='morgan', choices=['maccs', 'morgan'])
    parser.add_argument('--dosage_durations', type=int, nargs='+',
        default=[6, 24])
    parser.add_argument('--compound_concentrations', type=float, nargs='+',
        default=[9, 10, 11])
    parser.add_argument('--use_cd', action='store_true')
    parser.add_argument('--cell_line_consensus', action='store_true')

    # Training
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=4e-3)

    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_bins', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ctl_structure', type=str, default='SA+SA')
    parser.add_argument('--trt_structure', type=str, default='CA+SA+CA')
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--pcc_weight', type=float, default=1.0)
    parser.add_argument('--drug_hidden_dim', type=int, default=512)

    # Hyperparameter Search
    parser.add_argument('--hyperparameter_search', action='store_true', help='Run Optuna hyperparameter search before full CV')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')

    args = parser.parse_args()
    seq_pred_pipeline(args)