import re
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_ID = "2757168"
N_FOLDS = 5
OUT_FILE = os.path.join(LOG_DIR, f"training_summary_{JOB_ID}.png")

# ── Parsers ───────────────────────────────────────────────────────────────────
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+-\s+Train Loss:\s+([\d.]+)\s+-\s+Val Loss:\s+([\d.]+)\s+-\s+LR:\s+([\deE+\-.]+)"
)
METRIC_RE = {
    "pcc_mean": re.compile(r"Mean PCC:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "spearman": re.compile(r"Mean Spearman:\s+([\d.]+)\s+±\s+([\d.]+)"),
    "r2": re.compile(r"Mean R²:\s+([\d.]+)\s+±\s+([\d.]+)"),
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


# ── Load all folds ─────────────────────────────────────────────────────────────
fold_data = {}
for i in range(N_FOLDS):
    path = os.path.join(LOG_DIR, f"slurm-{JOB_ID}_{i}.out")
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    fold_data[i] = parse_log(path)
    print(f"Fold {i}: {len(fold_data[i][0])} epochs parsed, metrics: {list(fold_data[i][4].keys())}")

# ── Figure layout ──────────────────────────────────────────────────────────────
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f"Training Summary  –  Job {JOB_ID}", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Loss curves (one per fold, 5 subplots in first two rows) ───────────────
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

    # Annotate final val loss
    ax.annotate(
        f"Val: {val_loss[-1]:.4f}",
        xy=(epochs[-1], val_loss[-1]),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

# ── 2. LR schedule overlay (all folds on one plot) ────────────────────────────
ax_lr = fig.add_subplot(gs[1, 2])
for fold_idx, (epochs, _, _, lr, _) in sorted(fold_data.items()):
    ax_lr.plot(epochs, lr, label=f"Fold {fold_idx}", lw=1.2, alpha=0.8)
ax_lr.set_title("Learning Rate Schedule", fontsize=11)
ax_lr.set_xlabel("Epoch")
ax_lr.set_ylabel("LR")
ax_lr.legend(fontsize=7)
ax_lr.grid(True, alpha=0.3)

# ── 3. Final metrics bar chart ────────────────────────────────────────────────
metric_labels = {
    "pcc_mean": "PCC",
    "spearman": "Spearman",
    "r2": "R²",
    "rmse": "RMSE",
    "prec_up": "Prec@20 Up",
    "prec_down": "Prec@20 Down",
}

fold_ids = sorted(fold_data.keys())
x = np.arange(len(fold_ids))
width = 0.13

ax_bar = fig.add_subplot(gs[2, :])

for j, (key, label) in enumerate(metric_labels.items()):
    means = [fold_data[f][4].get(key, (0, 0))[0] for f in fold_ids]
    errs = [fold_data[f][4].get(key, (0, 0))[1] for f in fold_ids]
    bars = ax_bar.bar(
        x + j * width,
        means,
        width,
        yerr=errs,
        label=label,
        capsize=3,
        color=COLORS[j % len(COLORS)],
        alpha=0.85,
    )

ax_bar.set_xlabel("Fold")
ax_bar.set_ylabel("Score")
ax_bar.set_title("Test Metrics per Fold", fontsize=11)
ax_bar.set_xticks(x + width * (len(metric_labels) - 1) / 2)
ax_bar.set_xticklabels([f"Fold {f}" for f in fold_ids])
ax_bar.legend(fontsize=8, ncol=3)
ax_bar.grid(True, axis="y", alpha=0.3)
ax_bar.set_ylim(0, 1.05)

# ── Print cross-fold summary ───────────────────────────────────────────────────
print("\n── Cross-fold summary ──────────────────────────────────")
for key, label in metric_labels.items():
    vals = [fold_data[f][4].get(key, (float("nan"), 0))[0] for f in fold_ids]
    arr = np.array(vals)
    print(f"{label:15s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
