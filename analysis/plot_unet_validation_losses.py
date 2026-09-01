from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Input / output
# ------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent

IN_CSV = (
    ANALYSIS_DIR
    / "segmentation_validation_losses_5fold.csv"
)

OUT_PNG = (
    ANALYSIS_DIR
    / "segmentation_validation_loss_grid_5fold.png"
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(IN_CSV)

if df.empty:
    raise RuntimeError(f"No data found in {IN_CSV}")

if "fold" not in df.columns or "epoch" not in df.columns or "valid_combined_loss" not in df.columns:
    raise ValueError(
        "CSV must contain columns: fold, epoch, valid_combined_loss"
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
folds = sorted(df["fold"].unique())

fig, axes = plt.subplots(2, 5, figsize=(14, 7))
axes = axes.flatten()

for i, fold in enumerate(folds):
    ax = axes[i]

    g = df[df["fold"] == fold].copy()
    g = g.sort_values("epoch")

    ax.plot(
        g["epoch"],
        g["valid_combined_loss"],
        linewidth=1.8
    )

    # Display folds as 1..10 instead of 0..9
    ax.set_title(f"Fold {fold + 1}", fontsize=12)
    ax.set_xlabel("Epochs", fontsize=10)
    ax.set_ylabel("Validation Loss", fontsize=10)

    # Light grid, similar clean style
    ax.grid(alpha=0.25)

# If fewer than 10 folds, hide unused axes
for j in range(len(folds), len(axes)):
    axes[j].axis("off")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved plot to: {OUT_PNG}")