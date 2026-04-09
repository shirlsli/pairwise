import os
import pickle
import numpy as np
import argparse

def aggregate_results(args):
    fold_results = []
    for i in range(args.n_folds):
        results_path = os.path.join(args.model_output_dir, f'results_fold_{i}.pkl')
        with open(results_path, 'rb') as f:
            fold_results.append(pickle.load(f))

    pccs = [r['test_pcc'] for r in fold_results]
    spearmans = [r['test_spearman'] for r in fold_results]
    r2s = [r['test_r2'] for r in fold_results]
    rmses = [r['test_rmse'] for r in fold_results]

    print(f"\n{'='*50}")
    print(f"Cross-validation results ({len(fold_results)} folds):")
    print(f"Mean PCC: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}")
    print(f"Median PCC: {np.median(pccs):.4f}")
    print(f"Mean Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
    print(f"Median Spearman: {np.median(spearmans):.4f}")
    print(f"Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"Median R²: {np.median(r2s):.4f}")
    print(f"Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"Median RMSE: {np.median(rmses):.4f}")
    print(f"Mean Pos P@20: {np.mean([r['test_pos_p20'] for r in fold_results]):.4f} ± {np.std([r['test_pos_p20'] for r in fold_results]):.4f}")
    print(f"Median Pos P@20: {np.median([r['test_pos_p20'] for r in fold_results]):.4f}")
    print(f"Mean Neg P@20: {np.mean([r['test_neg_p20'] for r in fold_results]):.4f} ± {np.std([r['test_neg_p20'] for r in fold_results]):.4f}")
    print(f"Median Neg P@20: {np.median([r['test_neg_p20'] for r in fold_results]):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, required=True)
    args = parser.parse_args()
    aggregate_results(args)