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
    pos_p20s = [r['test_pos_p20'] for r in fold_results]
    neg_p20s = [r['test_neg_p20'] for r in fold_results]

    lines = [
        f"{'='*50}",
        f"Cross-validation results ({len(fold_results)} folds):",
        f"Mean PCC: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}",
        f"Median PCC: {np.median(pccs):.4f}",
        f"Mean Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}",
        f"Median Spearman: {np.median(spearmans):.4f}",
        f"Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}",
        f"Median R²: {np.median(r2s):.4f}",
        f"Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}",
        f"Median RMSE: {np.median(rmses):.4f}",
        f"Mean Pos P@20: {np.mean(pos_p20s):.4f} ± {np.std(pos_p20s):.4f}",
        f"Median Pos P@20: {np.median(pos_p20s):.4f}",
        f"Mean Neg P@20: {np.mean(neg_p20s):.4f} ± {np.std(neg_p20s):.4f}",
        f"Median Neg P@20: {np.median(neg_p20s):.4f}",
        f"{'='*50}",
    ]

    output = "\n".join(lines)
    print(output)

    output_path = os.path.join(args.model_output_dir, 'results_summary.txt')
    with open(output_path, 'w') as f:
        f.write(output + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, required=True)
    args = parser.parse_args()
    aggregate_results(args)