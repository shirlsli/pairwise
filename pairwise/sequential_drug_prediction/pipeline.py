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
from xpert_morgan import XPertMorgan, PerturbationDataset, evaluate_model, loss_fn
from hyperparameter_search import run_hyperparameter_search
import math
import json

def lr_lambda(epoch, warmup_epochs, args):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train_fold(split, args, device):
    fold_idx = split['fold']
    print(f"\n{'='*50}")
    print(f"Training fold {fold_idx}")
    print(f"{'='*50}")

    train_data = split['train']
    val_data = split['val']
    test_data = split['test']

    all_cell_ids = np.unique(np.concatenate([
        train_data['cell_ids'], val_data['cell_ids'], test_data['cell_ids']
    ]))
    cell_id_to_idx = {c: i for i, c in enumerate(all_cell_ids)}

    train_dataset = PerturbationDataset(train_data, cell_id_to_idx)
    val_dataset = PerturbationDataset(val_data,   cell_id_to_idx)
    test_dataset = PerturbationDataset(test_data,  cell_id_to_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False
    )

    config = {
        'dataset': {
            'gene_num': train_data['delta_expr'].shape[1],
            'n_bins': args.n_bins,
            'num_cell_id': len(cell_id_to_idx),
            'num_pert_time': 6,
        },
        'model': {
            'ATTN': {
                'hidden_size': args.hidden_size,
                'n_heads': args.num_heads,
                'attention_probs_dropout_prob': 0.1,
                'hidden_dropout_prob': 0.1,
                'cell_input_hidden_dropout_prob': 0.1,
                'drug_input_hidden_dropout_prob': 0.1,
                'ppi_gene_vector_path': args.ppi_gene_vector_path,
                'ctl_structure': args.ctl_structure,
                'trt_structure': args.trt_structure,
            },
        }
    }

    model_path = os.path.join(args.model_output_dir, f"model_fold_{fold_idx}.pt")
    checkpoint_path = os.path.join(args.model_output_dir, f"checkpoint_fold_{fold_idx}.pt")

    model = XPertMorgan(config, device)
    model.init_weights()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    warmup_epochs = 70
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, warmup_epochs, args)
    )

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    # resume from checkpoint if it exists and --resume is set
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    elif args.resume and os.path.exists(model_path):
        # partial resume — weights only, optimizer/scheduler restart
        print(f"No checkpoint found, resuming from weights only: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Training on {len(train_dataset)} samples...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            trt_output, ctl_output, deg_output, trt_raw, ctl_raw, \
                attn, cell_class_true, cell_class_predict = outputs
            loss = loss_fn(deg_output, trt_raw, cell_class_predict,
                           cell_class_true, args.mse_weight, args.pcc_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(batch)
                trt_output, ctl_output, deg_output, trt_raw, ctl_raw, \
                    attn, cell_class_true, cell_class_predict = outputs
                val_loss += loss_fn(deg_output, trt_raw, cell_class_predict,
                                    cell_class_true, args.mse_weight, args.pcc_weight).item()
        val_loss /= len(valid_loader)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model weights only
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

        # save full checkpoint every epoch for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    metrics = evaluate_model(model, test_loader, device)

    test_pcc = float(np.mean(metrics['pccs']))
    test_pcc_std = float(np.std(metrics['pccs']))
    test_spearman = float(np.mean(metrics['spearmans']))
    test_spearman_std = float(np.std(metrics['spearmans']))
    test_r2 = float(np.mean(metrics['r2s']))
    test_r2_std = float(np.std(metrics['r2s']))
    test_rmse = float(metrics['rmse'])

    print(f'Fold {fold_idx} test PCC: {test_pcc:.4f} ± {test_pcc_std:.4f}')
    print(f'Fold {fold_idx} test Spearman: {test_spearman:.4f} ± {test_spearman_std:.4f}')
    print(f'Fold {fold_idx} test R²: {test_r2:.4f} ± {test_r2_std:.4f}')
    print(f'Fold {fold_idx} test RMSE: {test_rmse:.4f}')
    print(f'Fold {fold_idx} test Most Upregulated 20 Genes Precision@20:   {np.mean(metrics["pos_p20"]):.4f} ± {np.std(metrics["pos_p20"]):.4f}')
    print(f'Fold {fold_idx} test Most Downregulated 20 Genes Precision@20: {np.mean(metrics["neg_p20"]):.4f} ± {np.std(metrics["neg_p20"]):.4f}')

    return {
        'fold': fold_idx,
        'test_pcc': test_pcc,
        'test_pcc_std': test_pcc_std,
        'test_spearman': test_spearman,
        'test_spearman_std': test_spearman_std,
        'test_r2': test_r2,
        'test_r2_std': test_r2_std,
        'test_rmse': test_rmse,
        'test_pos_p20': np.mean(metrics["pos_p20"]),
        'test_pos_p20_std': np.std(metrics["pos_p20"]),
        'test_neg_p20': np.mean(metrics["neg_p20"]),
        'test_neg_p20_std': np.std(metrics["neg_p20"]),
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

    fold_paths = [
        os.path.join(args.fold_output_dir, f"fold_{i}.pkl")
        for i in range(5)
    ]

    if all(os.path.exists(p) for p in fold_paths):
        print("Loading existing folds from disk...")
        all_splits = []
        for p in fold_paths:
            with open(p, 'rb') as f:
                all_splits.append(pickle.load(f))
        collapsed_path = args.processed_data_dir.replace('.pkl', '_collapsed.pkl')
    else:
        all_splits, collapsed_path = train_val_test_split(
            args.processed_data_dir, args, landmark_gene_ids
        )
        os.makedirs(args.fold_output_dir, exist_ok=True)
        for split in all_splits:
            fold_path = os.path.join(
                args.fold_output_dir, f"fold_{split['fold']}.pkl"
            )
            with open(fold_path, 'wb') as f:
                pickle.dump(split, f)
            print(f"Saved fold {split['fold']} to {fold_path}")

    if args.hyperparameter_search:
        print("Running hyperparameter search...")
        with open(args.config_path, 'r') as f:
            base_config = json.load(f)
        best_params = run_hyperparameter_search(
            all_splits, args.device, n_trials=args.n_trials,
            output_path='optuna_study.pkl', base_config=base_config
        )
        args.trt_structure = best_params['trt_structure']
        args.learning_rate = best_params['learning_rate']
        args.mse_weight = best_params['mse_weight']
        args.pcc_weight = best_params['pcc_weight']
        args.dropout = best_params['dropout']
        print(f"Best params found: {best_params}")

    if args.fold_idx is not None:
        all_splits = [s for s in all_splits if s['fold'] == args.fold_idx]
        print(f"Running fold {args.fold_idx} only")

    fold_results = []
    for split in all_splits:
        result = train_fold(split, args, device)
        fold_results.append(result)
        results_path = os.path.join(
            args.model_output_dir, f'results_fold_{result["fold"]}.pkl'
        )
        with open(results_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved fold {result['fold']} results to {results_path}")

    if args.fold_idx is None:
        pccs = [r['test_pcc']      for r in fold_results]
        spearmans = [r['test_spearman'] for r in fold_results]
        r2s       = [r['test_r2']       for r in fold_results]
        rmses     = [r['test_rmse']     for r in fold_results]

        print(f"\n{'='*50}")
        print(f"Cross-validation results ({len(pccs)} folds):")
        print(f"Mean PCC: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}")
        print(f"Min PCC: {np.min(pccs):.4f}")
        print(f"Max PCC: {np.max(pccs):.4f}")
        print(f"Mean Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
        print(f"Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        print(f"Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"Mean Pos P@20: {np.mean([r['test_pos_p20'] for r in fold_results]):.4f} ± {np.std([r['test_pos_p20'] for r in fold_results]):.4f}")
        print(f"Mean Neg P@20: {np.mean([r['test_neg_p20'] for r in fold_results]):.4f} ± {np.std([r['test_neg_p20'] for r in fold_results]):.4f}")
        print(f"{'='*50}")

    return fold_results

def inference(compounds_file_path):
    with open(compounds_file_path, 'r') as f:
        compounds = [line.strip() for line in f if line.strip()]
    drug_fps = preprocess_drugs(compounds)
    combos = {}
    for drug_fp in drug_fps:
        print(f"Predicting for drug fingerprint: {drug_fp}")
        baseline = cell_line_baselines[cell_id].astype(np.float32)
        delta_a = predict_delta(fp_a, baseline)
        state_after_a = baseline + delta_a

        for drug_b_fp in drug_fps:
            if drug_b_fp != drug_fp:
                delta_b = predict_delta(fp_b, state_after_a)
                drug_b_pert_id = drug_fps[drug_b_fp]['pert_id']
                state_after_b = state_after_a + delta_b
                combos[pert_id][drug_b_pert_id] = {
                    'baseline': baseline,
                    'delta_a': delta_a,
                    'state_after_a': state_after_a,
                    'delta_b': delta_b,
                    'state_after_b': state_after_b,
                    'total_delta': state_after_b - baseline,
                }
    # take each pair and call cell viability model to infer cell viability, then rank pairs by predicted viability

def predict_delta(fingerprint, current_expr):
    """Run one forward pass given a fingerprint and current expression state."""
    current_binned = apply_binning(
        current_expr.reshape(1, -1), bin_edges
    ).squeeze(0)

    batch = (
        torch.zeros(1, current_expr.shape[0], dtype=torch.float32).to(device),      # trt_raw_data placeholder
        torch.tensor(current_expr, dtype=torch.float32).unsqueeze(0).to(device),    # ctl_raw_data
        torch.tensor(current_binned, dtype=torch.long).unsqueeze(0).to(device),     # ctl_raw_data_binned
        torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).to(device),     # drug_feat
        torch.tensor([time_idx], dtype=torch.long).to(device),                      # pert_time_idx
        torch.tensor([cell_idx_val], dtype=torch.long).to(device),                  # cell_idx
    )

    with torch.no_grad():
        outputs = model(batch, output_attention=False)
        _, _, deg_output, _, _, _, _, _ = outputs

    return deg_output.squeeze(0).cpu().numpy()

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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=4e-3)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience in epochs')
    parser.add_argument('--warm_start', action='store_true', help='Whether to warm start, default is cold-drug training from scratch')
    parser.add_argument('--fold_idx', type=int, default=None, help='Fold index for parallelization')
    parser.add_argument('--resume', action='store_true', help='Resume training')

    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_bins', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ctl_structure', type=str, default='SA+SA+SA+SA')
    parser.add_argument('--trt_structure', type=str, default='CA+SA+SA+CA')
    parser.add_argument('--mse_weight', type=float, default=0.2)
    parser.add_argument('--pcc_weight', type=float, default=1.0)
    parser.add_argument('--drug_hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Config file
    parser.add_argument('--config_path', type=str, default='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/pairwise/sequential_drug_prediction/configs/config_lincs_l1000.json',
        help='Path to JSON config file; model/dataset values used as defaults, CLI args take precedence')

    # Hyperparameter Search
    parser.add_argument('--hyperparameter_search', action='store_true', help='Run Optuna hyperparameter search before full CV')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--ppi_gene_vector_path', type=str,
        default='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/PPI_gene_vector_128d.npy',
        help='Path to pretrained PPI gene vector embeddings')

    # First pass: get config_path only, then apply JSON values as defaults
    pre, _ = parser.parse_known_args()
    if pre.config_path:
        import json
        with open(pre.config_path) as f:
            cfg = json.load(f)
        attn = cfg.get('model', {}).get('ATTN', {})
        ds   = cfg.get('dataset', {})
        overrides = {}
        if 'n_bins' in ds:
            overrides['n_bins'] = ds['n_bins']
        if 'hidden_size' in attn:
            overrides['hidden_size'] = attn['hidden_size']
        if 'n_heads'in attn:
            overrides['num_heads'] = attn['n_heads']
        if 'ctl_structure' in attn:
            overrides['ctl_structure'] = attn['ctl_structure']
        if 'trt_structure' in attn:
            overrides['trt_structure'] = attn['trt_structure']
        if 'ppi_gene_vector_path' in attn:
            overrides['ppi_gene_vector_path'] = attn['ppi_gene_vector_path']
        parser.set_defaults(**overrides)

    args = parser.parse_args()
    seq_pred_pipeline(args)