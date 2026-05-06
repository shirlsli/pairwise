import os
import pickle
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

DATA = '/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl'
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cell_viability_regression')
os.makedirs(OUT, exist_ok=True)

ZS_ITERS  = 20
TEST_FRAC = 0.2
SEED      = 42
ALPHA     = 1.0
XGB_DEFAULTS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', verbosity=0)


def _load_xgb_best_params(gs_model_path):
    """Return best_params from a grid search bundle merged over XGB_DEFAULTS, or defaults alone."""
    if os.path.exists(gs_model_path):
        with open(gs_model_path, 'rb') as f:
            bundle = pickle.load(f)
        if 'best_params' in bundle:
            p = bundle['best_params']
            print(f"  Loaded best params from {os.path.basename(gs_model_path)}: {p}")
            return {**XGB_DEFAULTS, **p}
    print(f"  No grid search bundle at {gs_model_path}, using defaults.")
    return XGB_DEFAULTS


def _plot(Y, pred, title, out_file):
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
    ax.text(0.04, 0.96,
            f"PCC:      {pcc:.4f}\nSpearman: {spearman_r:.4f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_file}")


WARM_ITERS = 20


def _warm_start_loop(X, Y, make_model):
    np.random.seed(SEED)
    n = len(Y)
    train_n = n // 2
    pred_sums   = np.zeros(n, dtype=np.float64)
    pred_counts = np.zeros(n, dtype=np.int32)
    for i in range(WARM_ITERS):
        test_ind = np.random.choice(n, train_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_ind] = False
        print(f"  iter {i+1:2d}/{WARM_ITERS}  train={train_mask.sum()}  test={test_ind.shape[0]}")
        model = make_model(i)
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_ind]   += model.predict(X[test_ind])
        pred_counts[test_ind] += 1
    seen = pred_counts > 0
    return Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32)


def _zero_shot_loop(X, Y, cell_ids, drug_names, make_model):
    unique_cells = np.array(sorted(set(cell_ids)))
    np.random.seed(SEED)
    pred_sums   = np.zeros(len(Y), dtype=np.float64)
    pred_counts = np.zeros(len(Y), dtype=np.int32)
    for i in range(ZS_ITERS):
        n_test = max(1, int(len(unique_cells) * TEST_FRAC))
        test_cells  = set(np.random.choice(unique_cells, n_test, replace=False))
        train_cells = set(unique_cells) - test_cells
        train_mask  = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask   = np.array([c in test_cells and d in train_drugs
                                for c, d in zip(cell_ids, drug_names)])
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        print(f"  iter {i+1:2d}/{ZS_ITERS}  train={train_mask.sum()}  test={test_mask.sum()}")
        model = make_model(i)
        model.fit(X[train_mask], Y[train_mask])
        pred_sums[test_mask]   += model.predict(X[test_mask])
        pred_counts[test_mask] += 1
    seen = pred_counts > 0
    return Y[seen], (pred_sums[seen] / pred_counts[seen]).astype(np.float32)


print(f"Loading data from {DATA}...")
with open(DATA, 'rb') as f:
    data = pickle.load(f)

delta_expr   = data['delta_expr']
drug_fp      = data['drug_fp']
viability    = data['viability'].astype(np.float32)
cell_ids     = np.array(data['cell_ids'])
drug_names   = np.array(data['drug_names'])
duration_col_f32 = data['durations'].reshape(-1, 1).astype(np.float32)
duration_col_f64 = data['durations'].reshape(-1, 1).astype(np.float64)

X_fp_f64   = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float64)
X_expr_f64 = np.concatenate([delta_expr, duration_col_f64], axis=1).astype(np.float64)
X_fp_f32   = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float32)
X_expr_f32 = np.concatenate([delta_expr, duration_col_f32], axis=1).astype(np.float32)

xgb_fp_params   = _load_xgb_best_params(os.path.join(OUT, 'xgboost_model.pkl'))
xgb_expr_params = _load_xgb_best_params(os.path.join(OUT, 'xgb_gene_only_gs_model.pkl'))

# ── 1. Ridge zero-shot: Δexpr + FP ────────────────────────────────────────────
print("\n[1/4] Ridge zero-shot: Δexpr + FP")
Y_seen, pred = _zero_shot_loop(
    X_fp_f64, viability, cell_ids, drug_names,
    lambda i: Ridge(alpha=ALPHA, solver='svd'))
_plot(Y_seen, pred,
      f"Ridge Zero-Shot (Δexpr + FP): Predicted vs. Actual\n"
      f"({ZS_ITERS} iters, test_frac={TEST_FRAC}, alpha={ALPHA})",
      os.path.join(OUT, 'ridge_zero_shot_fp_predicted_vs_actual.png'))

# ── 2. Ridge zero-shot: Δexpr + duration ──────────────────────────────────────
print("\n[2/4] Ridge zero-shot: Δexpr + duration")
Y_seen, pred = _zero_shot_loop(
    X_expr_f64, viability, cell_ids, drug_names,
    lambda i: Ridge(alpha=ALPHA, solver='svd'))
_plot(Y_seen, pred,
      f"Ridge Zero-Shot (Δexpr + duration): Predicted vs. Actual\n"
      f"({ZS_ITERS} iters, test_frac={TEST_FRAC}, alpha={ALPHA})",
      os.path.join(OUT, 'ridge_zero_shot_expr_only_predicted_vs_actual.png'))

# ── 3. XGBoost zero-shot: Δexpr + FP ──────────────────────────────────────────
print("\n[3/4] XGBoost zero-shot: Δexpr + FP")
Y_seen, pred = _zero_shot_loop(
    X_fp_f32, viability, cell_ids, drug_names,
    lambda i: XGBRegressor(**{**xgb_fp_params, 'random_state': SEED + i}))
_plot(Y_seen, pred,
      f"XGBoost Zero-Shot (Δexpr + FP): Predicted vs. Actual\n"
      f"({ZS_ITERS} iters, test_frac={TEST_FRAC})",
      os.path.join(OUT, 'xgboost_zero_shot_fp_predicted_vs_actual.png'))

# ── 4. XGBoost zero-shot: Δexpr + duration ────────────────────────────────────
print("\n[4/4] XGBoost zero-shot: Δexpr + duration")
Y_seen, pred = _zero_shot_loop(
    X_expr_f32, viability, cell_ids, drug_names,
    lambda i: XGBRegressor(**{**xgb_expr_params, 'random_state': SEED + i}))
_plot(Y_seen, pred,
      f"XGBoost Zero-Shot (Δexpr + duration): Predicted vs. Actual\n"
      f"({ZS_ITERS} iters, test_frac={TEST_FRAC})",
      os.path.join(OUT, 'xgboost_zero_shot_expr_only_predicted_vs_actual.png'))

# ── 5. Ridge warm-start: Δexpr + FP ───────────────────────────────────────────
print("\n[5/8] Ridge warm-start: Δexpr + FP")
Y_seen, pred = _warm_start_loop(
    X_fp_f64, viability,
    lambda i: Ridge(alpha=ALPHA, solver='svd'))
_plot(Y_seen, pred,
      f"Ridge Warm-Start (Δexpr + FP): Predicted vs. Actual\n"
      f"({WARM_ITERS} iterations, alpha={ALPHA})",
      os.path.join(OUT, 'ridge_warm_start_fp_predicted_vs_actual.png'))

# ── 6. Ridge warm-start: Δexpr + duration ─────────────────────────────────────
print("\n[6/8] Ridge warm-start: Δexpr + duration")
Y_seen, pred = _warm_start_loop(
    X_expr_f64, viability,
    lambda i: Ridge(alpha=ALPHA, solver='svd'))
_plot(Y_seen, pred,
      f"Ridge Warm-Start (Δexpr + duration): Predicted vs. Actual\n"
      f"({WARM_ITERS} iterations, alpha={ALPHA})",
      os.path.join(OUT, 'ridge_warm_start_expr_only_predicted_vs_actual.png'))

# ── 7. XGBoost warm-start: Δexpr + FP ─────────────────────────────────────────
print("\n[7/8] XGBoost warm-start: Δexpr + FP")
Y_seen, pred = _warm_start_loop(
    X_fp_f32, viability,
    lambda i: XGBRegressor(**{**xgb_fp_params, 'random_state': SEED + i}))
_plot(Y_seen, pred,
      f"XGBoost Warm-Start (Δexpr + FP): Predicted vs. Actual\n"
      f"({WARM_ITERS} iterations)",
      os.path.join(OUT, 'xgboost_warm_start_fp_predicted_vs_actual.png'))

# ── 8. XGBoost warm-start: Δexpr + duration ───────────────────────────────────
print("\n[8/8] XGBoost warm-start: Δexpr + duration")
Y_seen, pred = _warm_start_loop(
    X_expr_f32, viability,
    lambda i: XGBRegressor(**{**xgb_expr_params, 'random_state': SEED + i}))
_plot(Y_seen, pred,
      f"XGBoost Warm-Start (Δexpr + duration): Predicted vs. Actual\n"
      f"({WARM_ITERS} iterations)",
      os.path.join(OUT, 'xgboost_warm_start_expr_only_predicted_vs_actual.png'))

print("\nDone.")
