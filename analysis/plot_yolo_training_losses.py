from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent

MODELS_DIR = (
    REPO_ROOT
    / "keypoints"
    / "models_5fold"
)

OUT_FILE = (
    ANALYSIS_DIR
    / "pose_box_loss_across_folds.png"
)

# Use only folders whose names start with fold_
result_files = sorted(MODELS_DIR.glob("fold*/results.csv"))



if not result_files:
    raise FileNotFoundError(f"No results.csv files found under {MODELS_DIR}/fold*/")

def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def fold_label(path):
    # path = models/fold_0/results.csv
    return path.parent.name

def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

fold_dfs = []

for results_csv in result_files:
    df = pd.read_csv(results_csv)
    df = clean_columns(df)

    if "epoch" not in df.columns:
        raise ValueError(f"No 'epoch' column in {results_csv}. Columns are: {list(df.columns)}")

    train_pose_col = get_col(df, ["train/pose_loss", "pose_loss"])
    val_pose_col = get_col(df, ["val/pose_loss"])

    train_box_col = get_col(df, ["train/box_loss", "box_loss"])
    val_box_col = get_col(df, ["val/box_loss"])

    fold_dfs.append({
        "fold": fold_label(results_csv),
        "df": df,
        "train_pose_col": train_pose_col,
        "val_pose_col": val_pose_col,
        "train_box_col": train_box_col,
        "val_box_col": val_box_col,
    })

print("Found folds:")
for item in fold_dfs:
    print(" ", item["fold"])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# ------------------------------------------------------------
# Pose loss
# ------------------------------------------------------------
ax = axes[0]

for item in fold_dfs:
    df = item["df"]

    if item["train_pose_col"] is not None:
        ax.plot(
            df["epoch"],
            df[item["train_pose_col"]],
            color="tab:blue",
            linestyle="-",
            alpha=0.65,
            linewidth=1.4,
        )

    if item["val_pose_col"] is not None:
        ax.plot(
            df["epoch"],
            df[item["val_pose_col"]],
            color="tab:orange",
            linestyle="--",
            alpha=0.65,
            linewidth=1.4,
        )

ax.set_title("Pose Loss Across Folds")
ax.set_xlabel("Epoch")
ax.set_ylabel("Pose Loss")
ax.grid(True, alpha=0.4)

# Dummy handles for clean legend
ax.plot([], [], color="tab:blue", linestyle="-", label="Train (solid lines)")
ax.plot([], [], color="tab:orange", linestyle="--", label="Validation (dotted lines)")
ax.legend()

# ------------------------------------------------------------
# Box loss
# ------------------------------------------------------------
ax = axes[1]

for item in fold_dfs:
    df = item["df"]

    if item["train_box_col"] is not None:
        ax.plot(
            df["epoch"],
            df[item["train_box_col"]],
            color="tab:blue",
            linestyle="-",
            alpha=0.65,
            linewidth=1.4,
        )

    if item["val_box_col"] is not None:
        ax.plot(
            df["epoch"],
            df[item["val_box_col"]],
            color="tab:orange",
            linestyle="--",
            alpha=0.65,
            linewidth=1.4,
        )

ax.set_title("Box Loss Across Folds")
ax.set_xlabel("Epoch")
ax.set_ylabel("Box Loss")
ax.grid(True, alpha=0.4)

ax.plot([], [], color="tab:blue", linestyle="-", label="Train (solid lines)")
ax.plot([], [], color="tab:orange", linestyle="--", label="Validation (dotted lines)")
ax.legend()

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300)
print(f"\nSaved plot to: {OUT_FILE.resolve()}")