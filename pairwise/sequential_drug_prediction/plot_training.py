import re
import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_ID = "2810626"
N_FOLDS = 5
OUT_FILE = os.path.join(LOG_DIR, f"training_summary_{JOB_ID}.png")

# ── Parsers ───────────────────────────────────────────────────────────────────
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+-\s+Train Loss:\s+([\d.]+)\s+-\s+Val Loss:\s+([\d.]+)\s+-\s+LR:\s+([\deE+\-.]+)"
)
METRIC_RE = {
    "pcc_mean": re.compile(r"Mean PCC:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "spearman": re.compile(r"Mean Spearman:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "r2": re.compile(r"Mean R²:\s+([-\d.]+)\s+±\s+([\d.]+)"),
    "rmse": re.compile(r"RMSE:\s+([\d.]+)"),
    "prec_up": re.compile(r"Mean Most Upregulated 20 Genes:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "prec_down": re.compile(r"Mean Most Downregulated 20 Genes:\s+([\d.]+)\s+±\s+([\d.]+)"),
}


def parse_log(path):
    epochs, train_loss, val_loss, lr = [], [], [], []
    metrics = {}
    with open(path) as f:
        text = f.read()

    for m in EPOCH_RE.finditer(text):
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        val_loss.append(float(m.group(3)))
        lr.append(float(m.group(4)))

    for key, pat in METRIC_RE.items():
        m = pat.search(text)
        if m:
            metrics[key] = (float(m.group(1)), float(m.group(2)) if pat.groups > 1 and len(m.groups()) > 1 else 0.0)

    # RMSE has only one group
    if "rmse" in metrics:
        m = METRIC_RE["rmse"].search(text)
        metrics["rmse"] = (float(m.group(1)), 0.0)

    return epochs, train_loss, val_loss, lr, metrics


def parse_optuna_journal(journal_path):
    """Parse an Optuna JournalFileBackend log into per-trial dicts."""
    params_raw    = {}
    dists         = {}
    states        = {}
    final_values  = {}
    intermediates = {}

    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            op  = entry.get("op_code")
            tid = entry.get("trial_id")
            if tid is None:
                continue

            if op == 5:
                params_raw.setdefault(tid, {})[entry["param_name"]] = entry["param_value_internal"]
                dists.setdefault(tid, {})[entry["param_name"]] = json.loads(entry["distribution"])
            elif op == 6:
                states[tid] = entry["state"]
                if entry.get("values"):
                    final_values[tid] = entry["values"][0]
            elif op == 7:
                v = entry["intermediate_value"]
                if v == v:  # skip NaN
                    intermediates.setdefault(tid, []).append((entry["step"], v))

    trials = {}
    for tid, raw in params_raw.items():
        decoded = {}
        for pname, pval in raw.items():
            d = dists[tid][pname]
            if d["name"] == "CategoricalDistribution":
                decoded[pname] = d["attributes"]["choices"][int(pval)]
            else:
                decoded[pname] = float(pval)
        obj = final_values.get(tid)
        trials[tid] = {
            "params":        decoded,
            "state":         states.get(tid, 0),
            "pcc":           -obj if obj is not None else None,
            "intermediates": sorted(intermediates.get(tid, []), key=lambda x: x[0]),
        }
    return trials


RIDGE_ITER_RE = re.compile(r"Iteration\s+(\d+)/\d+\s+PCC:\s+([\d.]+)")
RIDGE_SUMMARY_RE = {
    "mean": re.compile(r"Mean:\s+([\d.]+)"),
    "std":  re.compile(r"Std:\s+([\d.]+)"),
    "min":  re.compile(r"Min:\s+([\d.]+)"),
    "max":  re.compile(r"Max:\s+([\d.]+)"),
}


def parse_ridge_log(path):
    with open(path) as f:
        text = f.read()
    iterations = [(int(m.group(1)), float(m.group(2))) for m in RIDGE_ITER_RE.finditer(text)]
    summary = {k: float(pat.search(text).group(1)) for k, pat in RIDGE_SUMMARY_RE.items()
               if pat.search(text)}
    return iterations, summary


def plot_ridge_results(slurm_path, out_file=None):
    if out_file is None:
        base = os.path.splitext(slurm_path)[0]
        out_file = f"{base}_ridge_results.png"

    iterations, summary = parse_ridge_log(slurm_path)
    if not iterations:
        print(f"No ridge regression results found in {slurm_path}")
        return

    iters, pccs = zip(*iterations)
    mean, std = summary.get("mean", 0), summary.get("std", 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, pccs, marker="o", lw=1.5, ms=5, color="#1f77b4", label="PCC per fold")
    ax.axhline(mean, color="crimson", lw=1.5, ls="--", label=f"Mean = {mean:.4f}")
    ax.axhspan(mean - std, mean + std, alpha=0.12, color="crimson", label=f"±1 std ({std:.4f})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("PCC")
    ax.set_title(f"Ridge Regression {len(iters)}-Fold CV  –  {os.path.basename(slurm_path)}", fontsize=12)
    ax.set_xticks(list(iters))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    stats_text = (f"Mean: {mean:.4f}  Std: {std:.4f}\n"
                  f"Min: {summary.get('min', float('nan')):.4f}  "
                  f"Max: {summary.get('max', float('nan')):.4f}")
    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_file}")


def plot_zero_shot_predicted_vs_actual(data_pkl_path, alpha=1.0, zs_iters=20,
                                       test_frac=0.2, seed=42, out_file=None):
    from sklearn.linear_model import Ridge

    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_zero_shot_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    delta_expr = data['delta_expr']
    drug_fp    = data['drug_fp']
    viability  = data['viability']
    cell_ids   = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float64)
    Y = viability.astype(np.float32)

    unique_cell_lines = np.array(sorted(set(cell_ids)))
    np.random.seed(seed)

    pred_sums  = np.zeros(len(Y), dtype=np.float64)
    pred_counts = np.zeros(len(Y), dtype=np.int32)

    for _ in range(zs_iters):
        n_test_cells = max(1, int(len(unique_cell_lines) * test_frac))
        test_cells   = set(np.random.choice(unique_cell_lines, n_test_cells, replace=False))
        train_cells  = set(unique_cell_lines) - test_cells

        train_mask = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask  = np.array([c in test_cells and d in train_drugs
                                for c, d in zip(cell_ids, drug_names)])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = Ridge(alpha=alpha, solver='svd')
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_mask]   += model.predict(X[test_mask])
        pred_counts[test_mask] += 1

    seen = pred_counts > 0
    actual = Y[seen]
    pred   = (pred_sums[seen] / pred_counts[seen]).astype(np.float32)

    pcc = np.corrcoef(actual, pred)[0, 1]
    spearman_r, spearman_p = stats.spearmanr(actual, pred)
    slope, intercept, _, _, _ = stats.linregress(actual, pred)
    x_line = np.array([actual.min(), actual.max()])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, pred, alpha=0.15, s=4, color="#1f77b4", rasterized=True)
    ax.plot(x_line, x_line,                      lw=1.2, ls="--", color="gray")
    ax.plot(x_line, slope * x_line + intercept,  lw=1.5,          color="crimson")

    ax.set_xlabel("Actual Viability")
    ax.set_ylabel("Predicted Viability")
    ax.set_title(f"Ridge Regression Zero-Shot: Predicted vs. Actual\n"
                 f"({zs_iters} iterations, test_frac={test_frac}, alpha={alpha})", fontsize=11)
    ax.grid(True, alpha=0.3)

    stats_text = (f"PCC:      {pcc:.4f}\n"
                  f"Spearman: {spearman_r:.4f}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_file}")


def plot_ridge_predicted_vs_actual(model_pkl_path, data_pkl_path, out_file=None):
    """Ridge trained on [delta_expr + fp] — in-distribution."""
    if out_file is None:
        base = os.path.splitext(model_pkl_path)[0]
        out_file = f"{base}_predicted_vs_actual.png"

    with open(model_pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = bundle['model']
    X = np.concatenate([data['delta_expr'], data['drug_fp']], axis=1).astype(np.float64)
    Y = data['viability'].astype(np.float64)
    _make_predicted_vs_actual_plot(Y, model.predict(X),
                                   "Ridge (Δexpr + FP): Predicted vs. Actual", out_file)


def _make_predicted_vs_actual_plot(Y, pred, title, out_file):
    pcc = np.corrcoef(Y, pred)[0, 1]
    spearman_r, _ = stats.spearmanr(Y, pred)
    slope, intercept, _, _, _ = stats.linregress(Y, pred)
    x_line = np.array([Y.min(), Y.max()])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Y, pred, alpha=0.15, s=4, color="#1f77b4", rasterized=True)
    ax.plot(x_line, x_line, lw=1.2, ls="--", color="gray")
    ax.plot(x_line, slope * x_line + intercept, lw=1.5, color="crimson")
    ax.set_xlabel("Actual Viability")
    ax.set_ylabel("Predicted Viability")
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    stats_text = (f"PCC:      {pcc:.4f}\n"
                  f"Spearman: {spearman_r:.4f}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_file}")



def plot_ridge_expr_only_predicted_vs_actual(model_pkl_path, data_pkl_path, out_file=None):
    """Ridge trained on [delta_expr + duration] — gene-only, in-distribution."""
    if out_file is None:
        base = os.path.splitext(model_pkl_path)[0]
        out_file = f"{base}_predicted_vs_actual.png"

    with open(model_pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = bundle['model']
    duration_col = data['durations'].reshape(-1, 1).astype(np.float64)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float64)
    Y = data['viability'].astype(np.float64)
    _make_predicted_vs_actual_plot(Y, model.predict(X),
                                   "Ridge (Δexpr + duration): Predicted vs. Actual", out_file)


def plot_xgboost_expr_only_predicted_vs_actual(model_pkl_path, data_pkl_path, out_file=None):
    """XGBoost trained on [delta_expr + duration] — gene-only, in-distribution."""
    if out_file is None:
        base = os.path.splitext(model_pkl_path)[0]
        out_file = f"{base}_predicted_vs_actual.png"

    with open(model_pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = bundle['model']
    duration_col = data['durations'].reshape(-1, 1).astype(np.float32)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)
    _make_predicted_vs_actual_plot(Y, model.predict(X),
                                   "XGBoost (Δexpr + duration): Predicted vs. Actual", out_file)


def plot_zero_shot_expr_only_predicted_vs_actual(data_pkl_path, alpha=1.0, zs_iters=20,
                                                  test_frac=0.2, seed=42, out_file=None):
    """Ridge zero-shot cell-line on [delta_expr + duration]."""
    from sklearn.linear_model import Ridge
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_zero_shot_expr_only_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    duration_col = data['durations'].reshape(-1, 1).astype(np.float64)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float64)
    Y = data['viability'].astype(np.float32)
    cell_ids   = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    unique_cell_lines = np.array(sorted(set(cell_ids)))
    np.random.seed(seed)
    pred_sums   = np.zeros(len(Y), dtype=np.float64)
    pred_counts = np.zeros(len(Y), dtype=np.int32)

    for _ in range(zs_iters):
        n_test = max(1, int(len(unique_cell_lines) * test_frac))
        test_cells  = set(np.random.choice(unique_cell_lines, n_test, replace=False))
        train_cells = set(unique_cell_lines) - test_cells
        train_mask  = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask   = np.array([c in test_cells and d in train_drugs
                                for c, d in zip(cell_ids, drug_names)])
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = Ridge(alpha=alpha, solver='svd')
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_mask]   += model.predict(X[test_mask])
        pred_counts[test_mask] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"Ridge Zero-Shot (Δexpr + duration): Predicted vs. Actual\n"
        f"({zs_iters} iters, test_frac={test_frac}, alpha={alpha})", out_file)


def plot_zero_shot_xgboost_predicted_vs_actual(data_pkl_path, zs_iters=20, test_frac=0.2,
                                                seed=42, n_estimators=500, max_depth=6,
                                                learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, out_file=None):
    """XGBoost zero-shot cell-line on [delta_expr + fp]."""
    from xgboost import XGBRegressor
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_zero_shot_xgboost_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    X = np.concatenate([data['delta_expr'], data['drug_fp']], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)
    cell_ids   = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    unique_cell_lines = np.array(sorted(set(cell_ids)))
    np.random.seed(seed)
    pred_sums   = np.zeros(len(Y), dtype=np.float64)
    pred_counts = np.zeros(len(Y), dtype=np.int32)

    for i in range(zs_iters):
        n_test = max(1, int(len(unique_cell_lines) * test_frac))
        test_cells  = set(np.random.choice(unique_cell_lines, n_test, replace=False))
        train_cells = set(unique_cell_lines) - test_cells
        train_mask  = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask   = np.array([c in test_cells and d in train_drugs
                                for c, d in zip(cell_ids, drug_names)])
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, subsample=subsample,
                             colsample_bytree=colsample_bytree, tree_method='hist',
                             random_state=seed + i, verbosity=0)
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_mask]   += model.predict(X[test_mask])
        pred_counts[test_mask] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"XGBoost Zero-Shot (Δexpr + FP): Predicted vs. Actual\n"
        f"({zs_iters} iters, test_frac={test_frac})", out_file)


def plot_zero_shot_xgboost_expr_only_predicted_vs_actual(data_pkl_path, zs_iters=20,
                                                          test_frac=0.2, seed=42,
                                                          n_estimators=500, max_depth=6,
                                                          learning_rate=0.05, subsample=0.8,
                                                          colsample_bytree=0.8, out_file=None):
    """XGBoost zero-shot cell-line on [delta_expr + duration]."""
    from xgboost import XGBRegressor
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_zero_shot_xgboost_expr_only_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    duration_col = data['durations'].reshape(-1, 1).astype(np.float32)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)
    cell_ids   = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    unique_cell_lines = np.array(sorted(set(cell_ids)))
    np.random.seed(seed)
    pred_sums   = np.zeros(len(Y), dtype=np.float64)
    pred_counts = np.zeros(len(Y), dtype=np.int32)

    for i in range(zs_iters):
        n_test = max(1, int(len(unique_cell_lines) * test_frac))
        test_cells  = set(np.random.choice(unique_cell_lines, n_test, replace=False))
        train_cells = set(unique_cell_lines) - test_cells
        train_mask  = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask   = np.array([c in test_cells and d in train_drugs
                                for c, d in zip(cell_ids, drug_names)])
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, subsample=subsample,
                             colsample_bytree=colsample_bytree, tree_method='hist',
                             random_state=seed + i, verbosity=0)
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_mask]   += model.predict(X[test_mask])
        pred_counts[test_mask] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"XGBoost Zero-Shot (Δexpr + duration): Predicted vs. Actual\n"
        f"({zs_iters} iters, test_frac={test_frac})", out_file)


def plot_xgboost_predicted_vs_actual(model_pkl_path, data_pkl_path, out_file=None):
    """XGBoost trained on [delta_expr + fp] — in-distribution."""
    if out_file is None:
        base = os.path.splitext(model_pkl_path)[0]
        out_file = f"{base}_predicted_vs_actual.png"

    with open(model_pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    model = bundle['model']
    X = np.concatenate([data['delta_expr'], data['drug_fp']], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)
    _make_predicted_vs_actual_plot(Y, model.predict(X),
                                   "XGBoost (Δexpr + FP): Predicted vs. Actual", out_file)


_XGB_DEFAULTS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     tree_method='hist', verbosity=0)


def _load_xgb_best_params(gs_model_path):
    """Return best_params from a grid search bundle merged over defaults, or defaults alone."""
    if gs_model_path and os.path.exists(gs_model_path):
        with open(gs_model_path, 'rb') as f:
            bundle = pickle.load(f)
        if 'best_params' in bundle:
            p = bundle['best_params']
            print(f"  Loaded best params from {os.path.basename(gs_model_path)}: {p}")
            return {**_XGB_DEFAULTS, **p}
    if gs_model_path:
        print(f"  No grid search bundle at {gs_model_path}, using defaults.")
    return _XGB_DEFAULTS


def plot_warm_start_predicted_vs_actual(data_pkl_path, alpha=1.0, warm_iters=20,
                                        seed=42, out_file=None):
    """Ridge warm-start 50/50 CV on [delta_expr + fp]."""
    from sklearn.linear_model import Ridge
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_warm_start_ridge_fp_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    X = np.concatenate([data['delta_expr'], data['drug_fp']], axis=1).astype(np.float64)
    Y = data['viability'].astype(np.float32)

    np.random.seed(seed)
    n = len(Y)
    train_n = n // 2
    pred_sums   = np.zeros(n, dtype=np.float64)
    pred_counts = np.zeros(n, dtype=np.int32)

    for _ in range(warm_iters):
        test_ind = np.random.choice(n, train_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_ind] = False
        model = Ridge(alpha=alpha)
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_ind]   += model.predict(X[test_ind])
        pred_counts[test_ind] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"Ridge Warm-Start (Δexpr + FP): Predicted vs. Actual\n"
        f"({warm_iters} iterations, alpha={alpha})", out_file)


def plot_warm_start_expr_only_predicted_vs_actual(data_pkl_path, alpha=1.0, warm_iters=20,
                                                   seed=42, out_file=None):
    """Ridge warm-start 50/50 CV on [delta_expr + duration]."""
    from sklearn.linear_model import Ridge
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_warm_start_ridge_expr_only_predicted_vs_actual.png"

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    duration_col = data['durations'].reshape(-1, 1).astype(np.float64)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float64)
    Y = data['viability'].astype(np.float32)

    np.random.seed(seed)
    n = len(Y)
    train_n = n // 2
    pred_sums   = np.zeros(n, dtype=np.float64)
    pred_counts = np.zeros(n, dtype=np.int32)

    for _ in range(warm_iters):
        test_ind = np.random.choice(n, train_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_ind] = False
        model = Ridge(alpha=alpha, solver='svd')
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_ind]   += model.predict(X[test_ind])
        pred_counts[test_ind] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"Ridge Warm-Start (Δexpr + duration): Predicted vs. Actual\n"
        f"({warm_iters} iterations, alpha={alpha})", out_file)


def plot_warm_start_xgboost_predicted_vs_actual(data_pkl_path, warm_iters=20, seed=42,
                                                 gs_model_path=None, out_file=None):
    """XGBoost warm-start 50/50 CV on [delta_expr + fp]."""
    from xgboost import XGBRegressor
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_warm_start_xgboost_fp_predicted_vs_actual.png"

    params = _load_xgb_best_params(gs_model_path)

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    X = np.concatenate([data['delta_expr'], data['drug_fp']], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)

    np.random.seed(seed)
    n = len(Y)
    train_n = n // 2
    pred_sums   = np.zeros(n, dtype=np.float64)
    pred_counts = np.zeros(n, dtype=np.int32)

    for i in range(warm_iters):
        test_ind = np.random.choice(n, train_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_ind] = False
        model = XGBRegressor(**{**params, 'random_state': seed + i})
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_ind]   += model.predict(X[test_ind])
        pred_counts[test_ind] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"XGBoost Warm-Start (Δexpr + FP): Predicted vs. Actual\n"
        f"({warm_iters} iterations)", out_file)


def plot_warm_start_xgboost_expr_only_predicted_vs_actual(data_pkl_path, warm_iters=20,
                                                           seed=42, gs_model_path=None,
                                                           out_file=None):
    """XGBoost warm-start 50/50 CV on [delta_expr + duration]."""
    from xgboost import XGBRegressor
    if out_file is None:
        base = os.path.splitext(data_pkl_path)[0]
        out_file = f"{base}_warm_start_xgboost_expr_only_predicted_vs_actual.png"

    params = _load_xgb_best_params(gs_model_path)

    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    duration_col = data['durations'].reshape(-1, 1).astype(np.float32)
    X = np.concatenate([data['delta_expr'], duration_col], axis=1).astype(np.float32)
    Y = data['viability'].astype(np.float32)

    np.random.seed(seed)
    n = len(Y)
    train_n = n // 2
    pred_sums   = np.zeros(n, dtype=np.float64)
    pred_counts = np.zeros(n, dtype=np.int32)

    for i in range(warm_iters):
        test_ind = np.random.choice(n, train_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_ind] = False
        model = XGBRegressor(**{**params, 'random_state': seed + i})
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_ind]   += model.predict(X[test_ind])
        pred_counts[test_ind] += 1

    seen = pred_counts > 0
    _make_predicted_vs_actual_plot(
        Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32),
        f"XGBoost Warm-Start (Δexpr + duration): Predicted vs. Actual\n"
        f"({warm_iters} iterations)", out_file)


TRIAL_FINISH_RE = re.compile(
    r"Trial (\d+) finished with value: ([\-\d.]+) and parameters: (\{.*?\})"
)
HPSEARCH_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+-\s+Train Loss:\s+([\d.]+)\s+-\s+Val Loss:\s+([\d.]+)"
)
HPSEARCH_METRIC_RE = {
    "pcc":      re.compile(r"Mean PCC:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "spearman": re.compile(r"Mean Spearman:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "r2":       re.compile(r"Mean R²:\s+([-\d.]+)\s+±\s+([\d.]+)"),
    "rmse":     re.compile(r"RMSE:\s+([\d.]+)"),
    "prec_up":  re.compile(r"Mean Most Upregulated 20 Genes:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "prec_down":re.compile(r"Mean Most Downregulated 20 Genes:\s+([\d.]+)\s+±\s+([\d.]+)"),
}


def parse_hpsearch_slurm(path):
    """Parse a hyperparameter-search SLURM output into per-trial dicts."""
    with open(path) as f:
        text = f.read()

    # Split on trial-finish lines to get per-trial blocks
    boundaries = [m.start() for m in TRIAL_FINISH_RE.finditer(text)] + [len(text)]
    trials = {}
    for i, m in enumerate(TRIAL_FINISH_RE.finditer(text)):
        tid    = int(m.group(1))
        block  = text[boundaries[i - 1] if i > 0 else 0 : boundaries[i]]
        # epoch curves
        epochs, train_loss, val_loss = [], [], []
        for em in HPSEARCH_EPOCH_RE.finditer(block):
            epochs.append(int(em.group(1)))
            train_loss.append(float(em.group(2)))
            val_loss.append(float(em.group(3)))
        # metrics
        metrics = {}
        for key, pat in HPSEARCH_METRIC_RE.items():
            mm = pat.search(block)
            if mm:
                metrics[key] = (float(mm.group(1)),
                                float(mm.group(2)) if len(mm.groups()) > 1 else 0.0)
        # params (eval the dict string safely)
        import ast
        params = ast.literal_eval(m.group(3))
        trials[tid] = {
            "epochs":     epochs,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "metrics":    metrics,
            "params":     params,
            "pcc":        -float(m.group(2)),
        }
    return trials


def plot_optuna_trials(slurm_path, out_file=None, compare_ids=(0, 1)):
    """Compare two seeded trials from a hyperparameter-search SLURM output."""
    if out_file is None:
        base = os.path.splitext(slurm_path)[0]
        out_file = f"{base}_trial_comparison.png"

    trials = parse_hpsearch_slurm(slurm_path)
    pair   = [trials[i] for i in compare_ids]
    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange

    METRIC_LABELS = {
        "pcc": "PCC", "spearman": "Spearman", "r2": "R²",
        "rmse": "RMSE", "prec_up": "Prec@20 Up", "prec_down": "Prec@20 Down",
    }
    # Only show params that differ between the two trials
    all_params = list(pair[0]["params"].keys())
    diff_params = [
        p for p in all_params
        if pair[0]["params"].get(p) != pair[1]["params"].get(p)
    ]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Trial {compare_ids[0]} vs Trial {compare_ids[1]}  –  Initial Seeded Trials",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs_main = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)
    gs_top  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], wspace=0.35)
    gs_mid  = gridspec.GridSpecFromSubplotSpec(
        1, len(METRIC_LABELS), subplot_spec=gs_main[1], wspace=0.4
    )
    n_diff  = max(len(diff_params), 1)
    gs_bot  = gridspec.GridSpecFromSubplotSpec(1, n_diff, subplot_spec=gs_main[2], wspace=0.5)

    # ── 1. Val-loss convergence ───────────────────────────────────────────────
    ax_conv = fig.add_subplot(gs_top[0])
    for tid, t, color in zip(compare_ids, pair, colors):
        ax_conv.plot(t["epochs"], t["val_loss"], lw=1.8, color=color, label=f"Trial {tid}")
        ax_conv.plot(t["epochs"], t["train_loss"], lw=1.0, color=color,
                     ls="--", alpha=0.5)
    ax_conv.set_title("Loss Convergence  (solid=val, dashed=train)", fontsize=11)
    ax_conv.set_xlabel("Epoch")
    ax_conv.set_ylabel("Loss")
    ax_conv.legend(fontsize=9)
    ax_conv.grid(True, alpha=0.3)

    # ── 2. Final PCC bar ─────────────────────────────────────────────────────
    ax_pcc = fig.add_subplot(gs_top[1])
    for i, (tid, t, color) in enumerate(zip(compare_ids, pair, colors)):
        pcc = t["pcc"]
        ax_pcc.bar(i, pcc, color=color, width=0.4)
        ax_pcc.text(i, pcc + 0.003, f"{pcc:.4f}", ha="center", va="bottom", fontsize=10)
    ax_pcc.set_xticks([0, 1])
    ax_pcc.set_xticklabels([f"Trial {i}" for i in compare_ids])
    ax_pcc.set_title("Final PCC", fontsize=11)
    ax_pcc.set_ylabel("PCC")
    ax_pcc.grid(True, axis="y", alpha=0.3)

    # ── 3. All metrics side by side ───────────────────────────────────────────
    for col, (mkey, mlabel) in enumerate(METRIC_LABELS.items()):
        ax = fig.add_subplot(gs_mid[col])
        for i, (tid, t, color) in enumerate(zip(compare_ids, pair, colors)):
            val, err = t["metrics"].get(mkey, (0.0, 0.0))
            ax.bar(i, val, color=color, width=0.4, yerr=err, capsize=4)
            ax.text(i, val + err + 0.005, f"{val:.3f}", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"T{i}" for i in compare_ids], fontsize=8)
        ax.set_title(mlabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # ── 4. Differing parameters ───────────────────────────────────────────────
    for col, pname in enumerate(diff_params):
        ax = fig.add_subplot(gs_bot[col])
        for i, (tid, t, color) in enumerate(zip(compare_ids, pair, colors)):
            val = t["params"].get(pname)
            ax.bar(i, val, color=color, width=0.4)
            ax.text(i, val + abs(val) * 0.03, f"{val:.4g}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"T{i}" for i in compare_ids], fontsize=8)
        ax.set_title(pname.replace("_", " ").title(), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_file}")


def parse_slurm_array_trials(job_id, log_dir):
    """Parse one-trial-per-file SLURM array outputs into a dict keyed by array index."""
    import glob
    trials = {}
    for path in sorted(glob.glob(os.path.join(log_dir, f"slurm-{job_id}_*.out"))):
        idx = int(path.rsplit("_", 1)[-1].replace(".out", ""))
        parsed = parse_hpsearch_slurm(path)
        if parsed:
            trial = next(iter(parsed.values()))
            trials[idx] = trial
    return trials


def plot_journal_trials(job_id, log_dir, out_file=None):
    """Plot per-trial results parsed from SLURM array output files."""
    if out_file is None:
        out_file = os.path.join(log_dir, f"optuna_journal_optuna_summary.png")

    trials = parse_slurm_array_trials(job_id, log_dir)
    if not trials:
        print(f"No completed trial files found for job {job_id}.")
        return

    completed = {idx: t for idx, t in trials.items() if t.get("pcc") is not None}
    idxs = sorted(completed.keys())
    pccs = [completed[i]["pcc"] for i in idxs]

    PALETTE = [
        "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#17becf", "#ffbb78",
        "#aec7e8", "#f7b6d2", "#c5b0d5", "#c49c94",
    ]
    trial_color = {idx: PALETTE[i % len(PALETTE)] for i, idx in enumerate(sorted(trials.keys()))}

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Optuna Hyperparameter Search Summary  –  Job {job_id}",
                 fontsize=16, fontweight="bold", y=0.99)
    gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)

    # ── 1. Val-loss convergence ───────────────────────────────────────────────
    ax_conv = fig.add_subplot(gs_main[0])
    for idx, t in sorted(trials.items()):
        if not t["epochs"]:
            continue
        label = f"Worker {idx}" + ("" if t.get("pcc") is not None else " ✗")
        style = "-" if t.get("pcc") is not None else "--"
        ax_conv.plot(t["epochs"], t["val_loss"], lw=1.4, ls=style, alpha=0.85,
                     color=trial_color[idx], label=label)
    ax_conv.set_title("Val Loss Convergence per Trial  (dashed = failed)", fontsize=11)
    ax_conv.set_xlabel("Epoch")
    ax_conv.set_ylabel("Val Loss")
    ax_conv.legend(fontsize=7, ncol=10, loc="upper right")
    ax_conv.grid(True, alpha=0.3)

    # ── 2. PCC per completed trial ────────────────────────────────────────────
    ax_pcc = fig.add_subplot(gs_main[1])
    best_idx = int(np.argmax(pccs))
    for i, (idx, pcc) in enumerate(zip(idxs, pccs)):
        ec = "black" if i == best_idx else "none"
        lw = 1.5   if i == best_idx else 0
        ax_pcc.bar(i, pcc, color=trial_color[idx], edgecolor=ec, linewidth=lw)
    ax_pcc.set_xticks(range(len(idxs)))
    ax_pcc.set_xticklabels([f"W{i}" for i in idxs], fontsize=8)
    ax_pcc.set_title("PCC per Completed Trial", fontsize=11)
    ax_pcc.set_xlabel("Worker")
    ax_pcc.set_ylabel("PCC")
    ax_pcc.grid(True, axis="y", alpha=0.3)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_file}")


def plot_training_summary(job_id, log_dir, n_folds=5, out_file=None):
    """Plot per-fold loss curves, LR schedule, and metric bar chart for a training job."""
    if out_file is None:
        out_file = os.path.join(log_dir, f"training_summary_{job_id}.png")

    METRIC_LABELS = {
        "pcc_mean": "PCC",
        "spearman": "Spearman",
        "r2": "R²",
        "rmse": "RMSE",
        "prec_up": "Prec@20 Up",
        "prec_down": "Prec@20 Down",
    }
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fold_data = {}
    for i in range(n_folds):
        path = os.path.join(log_dir, f"slurm-{job_id}_{i}.out")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        fold_data[i] = parse_log(path)
        print(f"Fold {i}: {len(fold_data[i][0])} epochs parsed, metrics: {list(fold_data[i][4].keys())}")

    if not fold_data:
        print(f"No fold data found for job {job_id}.")
        return

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Training Summary  –  Job {job_id}", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Loss curves (one per fold) ────────────────────────────────────────────
    for i, (fold_idx, (epochs, train_loss, val_loss, lr, metrics)) in enumerate(sorted(fold_data.items())):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(epochs, train_loss, label="Train", color=COLORS[0], lw=1.5)
        ax.plot(epochs, val_loss, label="Val", color=COLORS[1], lw=1.5)
        ax.set_title(f"Fold {fold_idx}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.annotate(
            f"Val: {val_loss[-1]:.4f}",
            xy=(epochs[-1], val_loss[-1]),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    # ── LR schedule overlay ───────────────────────────────────────────────────
    ax_lr = fig.add_subplot(gs[1, 2])
    for fold_idx, (epochs, _, _, lr, _) in sorted(fold_data.items()):
        ax_lr.plot(epochs, lr, label=f"Fold {fold_idx}", lw=1.2, alpha=0.8)
    ax_lr.set_title("Learning Rate Schedule", fontsize=11)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.legend(fontsize=7)
    ax_lr.grid(True, alpha=0.3)

    # ── Final metrics bar chart ───────────────────────────────────────────────
    fold_ids = sorted(fold_data.keys())
    x = np.arange(len(fold_ids))
    width = 0.13
    ax_bar = fig.add_subplot(gs[2, :])
    for j, (key, label) in enumerate(METRIC_LABELS.items()):
        means = [fold_data[f][4].get(key, (0, 0))[0] for f in fold_ids]
        errs  = [fold_data[f][4].get(key, (0, 0))[1] for f in fold_ids]
        ax_bar.bar(x + j * width, means, width, yerr=errs, label=label,
                   capsize=3, color=COLORS[j % len(COLORS)], alpha=0.85)
    ax_bar.set_xlabel("Fold")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Test Metrics per Fold", fontsize=11)
    ax_bar.set_xticks(x + width * (len(METRIC_LABELS) - 1) / 2)
    ax_bar.set_xticklabels([f"Fold {f}" for f in fold_ids])
    ax_bar.legend(fontsize=8, ncol=3)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ── Cross-fold summary ────────────────────────────────────────────────────
    print(f"\n── Cross-fold summary  (Job {job_id}) ──────────────────────────────────")
    for key, label in METRIC_LABELS.items():
        vals = [fold_data[f][4].get(key, (float("nan"), 0))[0] for f in fold_ids]
        arr  = np.array(vals)
        print(f"{label:15s}: mean={np.nanmean(arr):.4f}  std={np.nanstd(arr):.4f}  "
              f"min={np.nanmin(arr):.4f}  max={np.nanmax(arr):.4f}")

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_file}")
    plt.close(fig)


plot_training_summary(JOB_ID, LOG_DIR, N_FOLDS)

# ── Seeded trial comparison ────────────────────────────────────────────────────
HPSEARCH_LOG = os.path.join(LOG_DIR, "slurm-2788447.out")
if os.path.exists(HPSEARCH_LOG):
    plot_optuna_trials(HPSEARCH_LOG)
else:
    print(f"Hyperparameter search log not found at {HPSEARCH_LOG}, skipping.")

# ── All optuna trials summary ─────────────────────────────────────────────────
HPSEARCH_JOB_ID = "2808941"
if any(
    os.path.exists(os.path.join(LOG_DIR, f"slurm-{HPSEARCH_JOB_ID}_{i}.out"))
    for i in range(20)
):
    plot_journal_trials(HPSEARCH_JOB_ID, LOG_DIR)
else:
    print(f"No SLURM array files found for job {HPSEARCH_JOB_ID}, skipping.")

plot_training_summary("2845224", LOG_DIR, n_folds=5)


CVR_DATA = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl'
CVR_OUT  = os.path.join(LOG_DIR, 'cell_viability_regression')

_models = {
    'ridge_fp':        ('ridge_model.pkl',              plot_ridge_predicted_vs_actual,              'ridge_fp_predicted_vs_actual.png'),
    'ridge_expr':      ('ridge_expr_only_model.pkl',    plot_ridge_expr_only_predicted_vs_actual,    'ridge_expr_only_predicted_vs_actual.png'),
    'xgboost_fp':      ('xgboost_model.pkl',            plot_xgboost_predicted_vs_actual,            'xgboost_fp_predicted_vs_actual.png'),
    'xgboost_expr':    ('xgboost_expr_only_model.pkl',  plot_xgboost_expr_only_predicted_vs_actual,  'xgboost_expr_only_predicted_vs_actual.png'),
}
for name, (model_file, fn, out_name) in _models.items():
    model_path = os.path.join(CVR_OUT, model_file)
    if os.path.exists(model_path) and os.path.exists(CVR_DATA):
        fn(model_pkl_path=model_path, data_pkl_path=CVR_DATA,
           out_file=os.path.join(CVR_OUT, out_name))
    else:
        print(f"{name} model not found at {model_path}, skipping.")

# ── Zero-shot predicted vs. actual ────────────────────────────────────────────
if os.path.exists(CVR_DATA):
    plot_zero_shot_predicted_vs_actual(
        data_pkl_path=CVR_DATA, alpha=1.0, zs_iters=20, test_frac=0.2, seed=42,
        out_file=os.path.join(CVR_OUT, 'ridge_zero_shot_fp_predicted_vs_actual.png'))
    plot_zero_shot_expr_only_predicted_vs_actual(
        data_pkl_path=CVR_DATA, alpha=1.0, zs_iters=20, test_frac=0.2, seed=42,
        out_file=os.path.join(CVR_OUT, 'ridge_zero_shot_expr_only_predicted_vs_actual.png'))
    plot_zero_shot_xgboost_predicted_vs_actual(
        data_pkl_path=CVR_DATA, zs_iters=20, test_frac=0.2, seed=42,
        out_file=os.path.join(CVR_OUT, 'xgboost_zero_shot_fp_predicted_vs_actual.png'))
    plot_zero_shot_xgboost_expr_only_predicted_vs_actual(
        data_pkl_path=CVR_DATA, zs_iters=20, test_frac=0.2, seed=42,
        out_file=os.path.join(CVR_OUT, 'xgboost_zero_shot_expr_only_predicted_vs_actual.png'))
else:
    print(f"CVR data not found at {CVR_DATA}, skipping zero-shot plots.")

# ── Warm-start predicted vs. actual ───────────────────────────────────────────
if os.path.exists(CVR_DATA):
    plot_warm_start_predicted_vs_actual(
        data_pkl_path=CVR_DATA, alpha=1.0, warm_iters=20, seed=42,
        out_file=os.path.join(CVR_OUT, 'ridge_warm_start_fp_predicted_vs_actual.png'))
    plot_warm_start_expr_only_predicted_vs_actual(
        data_pkl_path=CVR_DATA, alpha=1.0, warm_iters=20, seed=42,
        out_file=os.path.join(CVR_OUT, 'ridge_warm_start_expr_only_predicted_vs_actual.png'))
    plot_warm_start_xgboost_predicted_vs_actual(
        data_pkl_path=CVR_DATA, warm_iters=20, seed=42,
        gs_model_path=os.path.join(CVR_OUT, 'xgboost_model.pkl'),
        out_file=os.path.join(CVR_OUT, 'xgboost_warm_start_fp_predicted_vs_actual.png'))
    plot_warm_start_xgboost_expr_only_predicted_vs_actual(
        data_pkl_path=CVR_DATA, warm_iters=20, seed=42,
        gs_model_path=os.path.join(CVR_OUT, 'xgb_gene_only_gs_model.pkl'),
        out_file=os.path.join(CVR_OUT, 'xgboost_warm_start_expr_only_predicted_vs_actual.png'))
else:
    print(f"CVR data not found at {CVR_DATA}, skipping warm-start plots.")