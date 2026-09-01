from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ------------------------------------------------------------
# Paths / settings
# ------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DATA_DIR = REPO_ROOT / "data"

N_FOLDS = 5

# Change only this when comparing a different U-Net test run.
SEG_RESULTS_RUN = "0.9912_2026-08-16_results"

SEG_RESULTS = (
    REPO_ROOT
    / "segmentation"
    / SEG_RESULTS_RUN
    / "seg_tangent_sign"
)

KP_RESULTS = (
    REPO_ROOT
    / "keypoints"
    / "evaluation"
    / "results.csv"
)

SURFACE_OUT_CSV = (
    ANALYSIS_DIR
    / "unet_vs_keypoint_surface_difference.csv"
)

SURFACE_OUT_PNG = (
    ANALYSIS_DIR
    / "unet_vs_keypoint_surface_difference_scatter.png"
)

DICE_OUT_CSV = (
    ANALYSIS_DIR
    / "unet_vs_keypoint_dice.csv"
)

DICE_OUT_PNG = (
    ANALYSIS_DIR
    / "unet_vs_keypoint_dice_scatter.png"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_case_id(value):
    """
    Normalize filenames/case IDs.

    Examples
    --------
    A0001.jpg   -> A0001
    A0001_aug   -> A0001
    """
    value = Path(str(value).strip()).stem
    value = re.sub(r"_aug$", "", value)
    return value


def load_study_cases():
    """
    Load and validate the study cohort defined by the split CSVs.

    The split files are the single source of truth for cohort inclusion.
    Exclusions are applied upstream when create_splits.py creates them.
    """
    fold_case_sets = []

    for fold in range(N_FOLDS):
        split_path = (
            DATA_DIR
            / f"split_labels_fold_{fold}.csv"
        )

        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {split_path}"
            )

        split_df = pd.read_csv(split_path)

        if "patient" not in split_df.columns:
            raise ValueError(
                f"{split_path.name} does not contain a 'patient' column."
            )

        cases = {
            clean_case_id(value)
            for value in split_df["patient"].dropna()
        }

        fold_case_sets.append(cases)

    reference_cases = fold_case_sets[0]

    for fold, fold_cases in enumerate(
        fold_case_sets[1:],
        start=1,
    ):
        if fold_cases != reference_cases:
            missing = sorted(reference_cases - fold_cases)
            extra = sorted(fold_cases - reference_cases)

            raise ValueError(
                f"Fold {fold} does not contain the same study cohort as fold 0.\n"
                f"Missing from fold {fold}: {missing}\n"
                f"Extra in fold {fold}: {extra}"
            )

    return reference_cases


def load_segmentation_results():
    """Load per-case U-Net localization results."""
    if not SEG_RESULTS.exists():
        raise FileNotFoundError(
            f"Segmentation results folder not found: {SEG_RESULTS}"
        )

    rows = []

    for path in sorted(
        SEG_RESULTS.rglob("*_losses.csv")
    ):
        values = (
            pd.read_csv(path, header=None)
            .iloc[:, 0]
            .dropna()
            .tolist()
        )

        if len(values) < 3:
            print(
                f"Skipping malformed segmentation result: {path}"
            )
            continue

        rows.append(
            {
                "case_id": clean_case_id(values[0]),
                "unet_dice": float(values[1]),
                "unet_surface_difference_percent": float(values[2]),
                "seg_source": str(path),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No usable segmentation result files found in {SEG_RESULTS}"
        )

    duplicates = sorted(
        df.loc[
            df["case_id"].duplicated(keep=False),
            "case_id",
        ].unique()
    )

    if duplicates:
        raise ValueError(
            "Duplicate U-Net results found for: "
            + ", ".join(duplicates)
        )

    return df


def load_keypoint_results():
    """Load per-case YOLO keypoint localization results."""
    if not KP_RESULTS.exists():
        raise FileNotFoundError(
            f"Keypoint results file not found: {KP_RESULTS}"
        )

    df = pd.read_csv(KP_RESULTS)

    required_columns = {
        "pat",
        "dice",
        "m_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{KP_RESULTS.name} is missing columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["case_id"] = df["pat"].map(
        clean_case_id
    )

    df = df.rename(
        columns={
            "dice": "keypoint_dice",
            "m_score": "keypoint_m_score",
        }
    )

    # m_score is stored as a proportion; convert to percent
    # to match the U-Net surface-difference output.
    df["keypoint_surface_difference_percent"] = (
        df["keypoint_m_score"] * 100
    )

    df = df[
        [
            "case_id",
            "keypoint_dice",
            "keypoint_surface_difference_percent",
        ]
    ].copy()

    duplicates = sorted(
        df.loc[
            df["case_id"].duplicated(keep=False),
            "case_id",
        ].unique()
    )

    if duplicates:
        raise ValueError(
            "Duplicate keypoint results found for: "
            + ", ".join(duplicates)
        )

    return df


def pearson_summary(df, x_col, y_col, label):
    """Calculate and print Pearson correlation."""
    corr_df = df[
        [x_col, y_col]
    ].dropna()

    if len(corr_df) < 2:
        print(
            f"{label}: not enough observations "
            "to calculate Pearson correlation."
        )
        return None, None

    r, p = pearsonr(
        corr_df[x_col],
        corr_df[y_col],
    )

    print(
        f"{label}: "
        f"r={r:.3f}, p={p:.4g}, n={len(corr_df)}"
    )

    return r, p


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    study_cases = load_study_cases()

    print(
        f"Study cohort defined by split CSVs: "
        f"{len(study_cases)} cases"
    )

    seg = load_segmentation_results()
    kp = load_keypoint_results()

    # The split CSVs are authoritative for cohort membership.
    seg = seg.loc[
        seg["case_id"].isin(study_cases)
    ].copy()

    kp = kp.loc[
        kp["case_id"].isin(study_cases)
    ].copy()

    seg_cases = set(seg["case_id"])
    kp_cases = set(kp["case_id"])

    missing_unet = sorted(
        study_cases - seg_cases
    )

    missing_keypoint = sorted(
        study_cases - kp_cases
    )

    if missing_unet:
        print(
            f"\nStudy cases without U-Net localization result "
            f"({len(missing_unet)}):"
        )
        print(missing_unet)

    if missing_keypoint:
        print(
            f"\nStudy cases without keypoint localization result "
            f"({len(missing_keypoint)}):"
        )
        print(missing_keypoint)

    # Compare only cases for which both models have a result.
    df = seg.merge(
        kp,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(
        "case_id"
    ).reset_index(drop=True)

    print(
        f"\nMatched study cases with both model results: "
        f"{len(df)}"
    )

    if df.empty:
        raise ValueError(
            "No matched U-Net/keypoint results remain "
            "after filtering to the study cohort."
        )

    # --------------------------------------------------------
    # Surface difference comparison
    # --------------------------------------------------------
    surface_df = df[
        [
            "case_id",
            "unet_surface_difference_percent",
            "keypoint_surface_difference_percent",
            "seg_source",
        ]
    ].copy()

    surface_df.to_csv(
        SURFACE_OUT_CSV,
        index=False,
    )

    print(
        f"Saved surface-difference data: "
        f"{SURFACE_OUT_CSV}"
    )

    pearson_summary(
        surface_df,
        "keypoint_surface_difference_percent",
        "unet_surface_difference_percent",
        "Surface difference correlation",
    )

    plt.figure(figsize=(8, 6))

    colors = range(len(surface_df))

    plt.scatter(
        surface_df[
            "keypoint_surface_difference_percent"
        ],
        surface_df[
            "unet_surface_difference_percent"
        ],
        c=colors,
        cmap="tab20",
        alpha=1,
        s=45,
    )

    max_val = max(
        100,
        surface_df[
            "keypoint_surface_difference_percent"
        ].max(),
        surface_df[
            "unet_surface_difference_percent"
        ].max(),
    )

    plt.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        color="gray",
        linewidth=1.5,
    )

    plt.xlabel(
        "YOLO keypoint model surface difference (%)"
    )
    plt.ylabel(
        "U-Net surface difference (%)"
    )
    plt.title(
        "U-Net vs YOLO keypoint surface difference"
    )

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        SURFACE_OUT_PNG,
        dpi=300,
    )
    plt.close()

    print(
        f"Saved surface-difference plot: "
        f"{SURFACE_OUT_PNG}"
    )

    # --------------------------------------------------------
    # Dice comparison
    # --------------------------------------------------------
    dice_df = df[
        [
            "case_id",
            "unet_dice",
            "keypoint_dice",
            "seg_source",
        ]
    ].copy()

    dice_df.to_csv(
        DICE_OUT_CSV,
        index=False,
    )

    print(
        f"\nSaved Dice data: {DICE_OUT_CSV}"
    )

    print("\nDice summary:")
    print(
        dice_df[
            ["unet_dice", "keypoint_dice"]
        ].describe()
    )

    pearson_summary(
        dice_df,
        "keypoint_dice",
        "unet_dice",
        "Dice correlation",
    )

    plt.figure(figsize=(8, 6))

    colors = range(len(dice_df))

    plt.scatter(
        dice_df["keypoint_dice"],
        dice_df["unet_dice"],
        c=colors,
        cmap="tab20",
        alpha=1,
        s=45,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
    )

    plt.xlabel(
        "YOLO keypoint model Dice"
    )
    plt.ylabel(
        "U-Net Dice"
    )
    plt.title(
        "U-Net vs YOLO keypoint Dice"
    )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(
        DICE_OUT_PNG,
        dpi=300,
    )
    plt.close()

    print(
        f"Saved Dice plot: "
        f"{DICE_OUT_PNG}"
    )


if __name__ == "__main__":
    main()
