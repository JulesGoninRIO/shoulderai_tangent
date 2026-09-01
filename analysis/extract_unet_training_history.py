from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt
import torch

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2


# ------------------------------------------------------------
# Paths / selected W&B runs
# ------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
SEG_DIR = REPO_ROOT / "segmentation"

WANDB_DIR = SEG_DIR / "wandb"
CHECK_DIR = SEG_DIR / "check"

OUT_CSV = (
    ANALYSIS_DIR
    / "segmentation_validation_losses_5fold.csv"
)

OUT_MATCHES = (
    ANALYSIS_DIR
    / "segmentation_wandb_checkpoint_matches_5fold.csv"
)

OUT_PNG = (
    ANALYSIS_DIR
    / "segmentation_validation_loss_5fold.png"
)

MATCH_TOL = 1e-5
N_FOLDS = 5

SELECT_RUN_DIRS = [
    "offline-run-20260814_134714-seg_tangent_sign_test",
]

# ------------------------------------------------------------
# W&B parsing
# ------------------------------------------------------------
def get_history_key(item):
    nested = list(getattr(item, "nested_key", []))
    if nested:
        return "/".join(nested)
    return getattr(item, "key", "")


def parse_wandb_file(wandb_file: Path) -> pd.DataFrame:
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))

    rows = []

    while True:
        data = ds.scan_data()
        if data is None:
            break

        if isinstance(data, tuple):
            data = data[-1]

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)

        if record.HasField("history"):
            row = {}

            for item in record.history.item:
                key = get_history_key(item)

                if not key:
                    continue

                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json

            if row:
                rows.append(row)

    return pd.DataFrame(rows)


def find_column(df, terms):
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if all(t in lc for t in terms):
            candidates.append(c)
    return candidates[0] if candidates else None


def split_blocks_by_epoch_reset(df: pd.DataFrame, epoch_col: str, val_col: str):
    d = df.dropna(subset=[epoch_col, val_col]).copy()

    if d.empty:
        return []

    if "_step" in d.columns:
        d = d.sort_values("_step")

    d = d.reset_index(drop=True)

    blocks = []
    current = []
    prev_epoch = None

    for _, row in d.iterrows():
        epoch = row[epoch_col]

        if prev_epoch is not None and epoch < prev_epoch:
            blocks.append(pd.DataFrame(current))
            current = []

        current.append(row)
        prev_epoch = epoch

    if current:
        blocks.append(pd.DataFrame(current))

    return blocks


# ------------------------------------------------------------
# Checkpoint parsing
# ------------------------------------------------------------
def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_best_score(ckpt):
    for _, callback_state in ckpt.get("callbacks", {}).items():
        if not isinstance(callback_state, dict):
            continue

        for key, value in callback_state.items():
            if "best_model_score" in key:
                try:
                    return float(value)
                except Exception:
                    return float(value.item())

    return None


def load_checkpoint_table():
    rows = []

    for ckpt_path in sorted(CHECK_DIR.glob("*.ckpt")):
        m_fold = re.search(r"fold[_=-]?(\d+)", ckpt_path.name)
        m_epoch = re.search(r"epoch[=_-]?(\d+)", ckpt_path.name)

        if not m_fold:
            continue

        ckpt = torch_load_checkpoint(ckpt_path)

        rows.append({
            "fold": int(m_fold.group(1)),
            "checkpoint_epoch": int(m_epoch.group(1)) if m_epoch else ckpt.get("epoch"),
            "checkpoint_best_score": extract_best_score(ckpt),
            "checkpoint_file": ckpt_path.name,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No checkpoint files found in {CHECK_DIR}")

    if df["checkpoint_best_score"].isna().any():
        print("Warning: some checkpoints have no best_model_score.")

    return df.sort_values("fold").reset_index(drop=True)


def get_block_timing(block: pd.DataFrame):
    """
    Estimate the training duration of one fold from W&B validation timestamps.

    Validation loss is logged at the end of each epoch. Therefore:
    - first_val_timestamp = end of first epoch
    - last_val_timestamp = end of final epoch
    - observed span misses approximately one epoch
    - estimated total adds the median epoch duration once
    """

    if "_timestamp" not in block.columns:
        return {
            "first_val_timestamp": None,
            "last_val_timestamp": None,
            "observed_span_seconds": None,
            "median_epoch_seconds": None,
            "estimated_training_seconds": None,
        }

    b = block.dropna(subset=["_timestamp", "epoch"]).copy()

    if b.empty:
        return {
            "first_val_timestamp": None,
            "last_val_timestamp": None,
            "observed_span_seconds": None,
            "median_epoch_seconds": None,
            "estimated_training_seconds": None,
        }

    # One timestamp per epoch: validation is logged at the end of the epoch
    epoch_times = (
        b.groupby("epoch")["_timestamp"]
        .max()
        .sort_index()
    )

    first_ts = float(epoch_times.iloc[0])
    last_ts = float(epoch_times.iloc[-1])

    observed_span = last_ts - first_ts

    intervals = epoch_times.diff().dropna()

    if len(intervals) > 0:
        median_epoch_seconds = float(intervals.median())

        # first timestamp is already at the END of epoch 0,
        # so add one typical epoch to estimate complete training time
        estimated_training_seconds = (
            observed_span + median_epoch_seconds
        )
    else:
        median_epoch_seconds = None
        estimated_training_seconds = None

    return {
        "first_val_timestamp": first_ts,
        "last_val_timestamp": last_ts,
        "observed_span_seconds": observed_span,
        "median_epoch_seconds": median_epoch_seconds,
        "estimated_training_seconds": estimated_training_seconds,
    }

# ------------------------------------------------------------
# Build W&B block table
# ------------------------------------------------------------
def load_wandb_blocks():
    block_rows = []
    block_data = {}

    for run_name in SELECT_RUN_DIRS:
        run_dir = WANDB_DIR / run_name
        wandb_files = list(run_dir.glob("*.wandb"))

        if not wandb_files:
            raise FileNotFoundError(f"No .wandb file found in {run_dir}")

        df = parse_wandb_file(wandb_files[0])

        if df.empty:
            print(f"Warning: no history rows in {run_name}")
            continue

        epoch_col = find_column(df, ["epoch"])
        val_col = find_column(df, ["valid", "loss"])
        train_col = find_column(df, ["train", "loss"])

        if epoch_col is None or val_col is None:
            print(f"Skipping {run_name}: no epoch/validation-loss column found")
            print("Columns:", list(df.columns))
            continue

        blocks = split_blocks_by_epoch_reset(df, epoch_col, val_col)

        for block_idx, block in enumerate(blocks):
            if block.empty:
                continue

            block = block.copy()

            # Standardize columns
            keep = [epoch_col, val_col]
            rename = {
                epoch_col: "epoch",
                val_col: "valid_combined_loss",
            }

            if "_step" in block.columns:
                keep.append("_step")

            if "_timestamp" in block.columns:
                keep.append("_timestamp")

            if "_runtime" in block.columns:
                keep.append("_runtime")

            if train_col is not None and train_col in block.columns:
                keep.append(train_col)
                rename[train_col] = "train_combined_loss"

            block = block[keep].rename(columns=rename)
            block = block.dropna(subset=["valid_combined_loss"]).copy()

            if block.empty:
                continue

            key = (run_name, block_idx)
            block_data[key] = block

            timing = get_block_timing(block)

            block_rows.append({
                "run_dir": run_name,
                "block": block_idx,
                "n_rows": len(block),
                "first_epoch": float(block["epoch"].min()),
                "last_epoch": float(block["epoch"].max()),
                "min_val_loss": float(block["valid_combined_loss"].min()),

                "first_val_timestamp": timing["first_val_timestamp"],
                "last_val_timestamp": timing["last_val_timestamp"],
                "observed_span_seconds": timing["observed_span_seconds"],
                "median_epoch_seconds": timing["median_epoch_seconds"],
                "estimated_training_seconds": timing["estimated_training_seconds"],
            })

    blocks_df = pd.DataFrame(block_rows)

    if blocks_df.empty:
        raise RuntimeError("No usable W&B validation-loss blocks found.")

    return blocks_df, block_data


# ------------------------------------------------------------
# Match W&B blocks to checkpoint folds
# ------------------------------------------------------------
def match_blocks_to_folds(checkpoints, blocks_df):
    blocks_df = blocks_df.sort_values(
        ["run_dir", "block"]
    ).reset_index(drop=True)

    if len(blocks_df) != N_FOLDS:
        raise RuntimeError(
            f"Expected {N_FOLDS} W&B blocks, found {len(blocks_df)}"
        )

    matches = []

    for fold in range(N_FOLDS):
        block = blocks_df.iloc[fold]

        ckpt = checkpoints[
            checkpoints["fold"] == fold
        ]

        if len(ckpt) != 1:
            raise RuntimeError(
                f"Expected exactly one checkpoint for fold {fold}, "
                f"found {len(ckpt)}"
            )

        ckpt = ckpt.iloc[0]

        matches.append({
            "fold": fold,
            "checkpoint_epoch": ckpt["checkpoint_epoch"],
            "checkpoint_best_score": ckpt["checkpoint_best_score"],
            "checkpoint_file": ckpt["checkpoint_file"],
            "run_dir": block["run_dir"],
            "block": int(block["block"]),
            "block_last_epoch": block["last_epoch"],
            "block_min_val_loss": block["min_val_loss"],

            "median_epoch_seconds": block["median_epoch_seconds"],
            "estimated_training_seconds": block["estimated_training_seconds"],
            "estimated_training_minutes": block["estimated_training_seconds"] / 60,
            "estimated_training_hours": block["estimated_training_seconds"] / 3600,

            "abs_diff": abs(
                block["min_val_loss"]
                - ckpt["checkpoint_best_score"]
            ),
        })

    return pd.DataFrame(matches)


# ------------------------------------------------------------
# Combine curves and plot
# ------------------------------------------------------------
def build_combined_curves(matches, block_data):
    dfs = []

    for _, row in matches.iterrows():
        key = (row["run_dir"], int(row["block"]))

        if key not in block_data:
            print(f"Missing block data for {key}")
            continue

        block = block_data[key].copy()
        block["fold"] = int(row["fold"])
        block["checkpoint_epoch"] = row["checkpoint_epoch"]
        block["checkpoint_best_score"] = row["checkpoint_best_score"]
        block["run_dir"] = row["run_dir"]
        block["block"] = int(row["block"])

        dfs.append(block)

    if not dfs:
        raise RuntimeError("No fold curves could be combined.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["fold", "epoch"]).reset_index(drop=True)

    return combined


def plot_curves(combined, matches):
    plt.figure(figsize=(11, 6))

    for fold, g in combined.groupby("fold"):
        g = g.sort_values("epoch")

        plt.plot(
            g["epoch"],
            g["valid_combined_loss"],
            linewidth=1.5,
            alpha=0.85,
            label=f"Fold {fold}",
        )

        # Mark best checkpoint epoch
        m = matches[matches["fold"] == fold].iloc[0]
        plt.scatter(
            m["checkpoint_epoch"],
            m["checkpoint_best_score"],
            s=25,
            zorder=5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation combined loss")
    plt.title("Segmentation validation loss across folds")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)

    print(f"Saved plot: {OUT_PNG}")


def main():
    checkpoints = load_checkpoint_table()
    blocks_df, block_data = load_wandb_blocks()
    matches = match_blocks_to_folds(checkpoints, blocks_df)

    matches.to_csv(OUT_MATCHES, index=False)
    print("\nMatched folds:")
    print(matches.to_string(index=False))
    print(f"\nSaved matches: {OUT_MATCHES}")


    print("\nEstimated U-Net training times:")
    print(
        matches[
            [
                "fold",
                "checkpoint_epoch",
                "block_last_epoch",
                "median_epoch_seconds",
                "estimated_training_minutes",
                "estimated_training_hours",
            ]
        ].to_string(index=False)
    )

    total_seconds = matches["estimated_training_seconds"].sum()

    print(
        f"\nTotal estimated 5-fold training time: "
        f"{total_seconds / 3600:.2f} hours"
    )

    print(
        f"Mean estimated training time per fold: "
        f"{matches['estimated_training_seconds'].mean() / 60:.1f} minutes"
    )

    combined = build_combined_curves(matches, block_data)
    combined.to_csv(OUT_CSV, index=False)
    print(f"Saved combined validation losses: {OUT_CSV}")

    plot_curves(combined, matches)


if __name__ == "__main__":
    main()