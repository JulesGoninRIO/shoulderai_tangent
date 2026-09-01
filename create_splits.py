from pathlib import Path
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

EXPERT_CSV = DATA_DIR / "tangent_sign_expert_annotation.csv"
EXCLUSIONS_CSV = DATA_DIR / "cohort_exclusions.csv"

OUTPUT_DIR = DATA_DIR

N_FOLDS = 5
SEED = 2026




# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def base_patient_id(case_id):
    """
    F0051-2 -> F0051
    F0051-3 -> F0051
    A0012   -> A0012
    """
    return re.sub(r"-\d+$", "", case_id)


def load_exclusions(path: Path):
    """Load exact-case and base-patient exclusions."""

    if not path.exists():
        raise FileNotFoundError(
            f"Exclusion file not found: {path}"
        )

    exclusions = pd.read_csv(path)

    required_columns = {
        "case_id",
        "scope",
    }

    missing_columns = (
        required_columns - set(exclusions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Exclusion file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    exclusions["case_id"] = (
        exclusions["case_id"]
        .astype(str)
        .str.strip()
        .map(lambda value: Path(value).stem)
    )

    exclusions["scope"] = (
        exclusions["scope"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_scopes = set(
        exclusions.loc[
            ~exclusions["scope"].isin(
                {"exact", "base"}
            ),
            "scope",
        ]
    )

    if invalid_scopes:
        raise ValueError(
            "Invalid exclusion scopes: "
            + ", ".join(sorted(invalid_scopes))
        )

    exact_cases = set(
        exclusions.loc[
            exclusions["scope"] == "exact",
            "case_id",
        ]
    )

    base_patients = set(
        exclusions.loc[
            exclusions["scope"] == "base",
            "case_id",
        ]
    )

    return exact_cases, base_patients

# ---------------------------------------------------------------------
# Read expert annotations
# ---------------------------------------------------------------------

df = pd.read_csv(
    EXPERT_CSV,
    sep=None,
    engine="python",
)

df = df[
    ["image_file", "tangent_sign_numeric"]
].dropna().copy()


df["patient"] = df["image_file"].map(
    lambda x: Path(str(x)).stem
)

# Apply exclusions from cohort_exclusions.csv
excluded_exact_cases, excluded_base_patients = load_exclusions(
    EXCLUSIONS_CSV
)

exclusion_mask = df["patient"].map(
    lambda case_id: (
        case_id in excluded_exact_cases
        or base_patient_id(case_id) in excluded_base_patients
    )
)

df = df.loc[~exclusion_mask].copy()

# If the same image occurs several times, keep it once
df = df.drop_duplicates(
    subset="patient"
).reset_index(drop=True)

# Group repeated scans from the same patient
df["group"] = df["patient"].map(base_patient_id)


# ---------------------------------------------------------------------
# Dataset information
# ---------------------------------------------------------------------

print(f"Total cases: {len(df)}")

print("\nOverall class distribution:")
print(
    df["tangent_sign_numeric"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------------------
# Build patient groups
# ---------------------------------------------------------------------

groups = []

for group_id, group_df in df.groupby("group"):

    groups.append(
        {
            "group": group_id,
            "n_positive": int(
                (group_df["tangent_sign_numeric"] == 1).sum()
            ),
            "n_total": len(group_df),
        }
    )


# ---------------------------------------------------------------------
# Initialize folds
# ---------------------------------------------------------------------

folds = [
    {
        "groups": [],
        "n_positive": 0,
        "n_total": 0,
    }
    for _ in range(N_FOLDS)
]

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------
# 1. Distribute positive groups first
# ---------------------------------------------------------------------

positive_groups = [
    g for g in groups
    if g["n_positive"] > 0
]

# Randomize ties, then place groups with most positives first
rng.shuffle(positive_groups)

positive_groups = sorted(
    positive_groups,
    key=lambda g: g["n_positive"],
    reverse=True,
)

for group in positive_groups:

    # Prefer fold with fewest positives;
    # if tied, prefer smallest fold
    fold = min(
        folds,
        key=lambda f: (
            f["n_positive"],
            f["n_total"],
        ),
    )

    fold["groups"].append(group["group"])
    fold["n_positive"] += group["n_positive"]
    fold["n_total"] += group["n_total"]


# ---------------------------------------------------------------------
# 2. Distribute negative-only groups to balance fold size
# ---------------------------------------------------------------------

negative_groups = [
    g for g in groups
    if g["n_positive"] == 0
]

rng.shuffle(negative_groups)

# Larger groups first makes balancing easier
negative_groups = sorted(
    negative_groups,
    key=lambda g: g["n_total"],
    reverse=True,
)

for group in negative_groups:

    fold = min(
        folds,
        key=lambda f: f["n_total"],
    )

    fold["groups"].append(group["group"])
    fold["n_total"] += group["n_total"]


# ---------------------------------------------------------------------
# Save splits
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for fold_idx, fold in enumerate(folds):

    test_groups = set(fold["groups"])

    out = df[["patient"]].copy()

    out["phase"] = np.where(
        df["group"].isin(test_groups),
        "test",
        "train",
    )

    output_file = (
        OUTPUT_DIR
        / f"split_labels_fold_{fold_idx}.csv"
    )

    out.to_csv(
        output_file,
        index=False,
    )

    # -------------------------------------------------------------
    # Print fold statistics
    # -------------------------------------------------------------

    test_df = df[
        df["group"].isin(test_groups)
    ]

    train_df = df[
        ~df["group"].isin(test_groups)
    ]

    print(f"\nFold {fold_idx}")
    print(
        f"Train: {len(train_df)} | "
        f"Test: {len(test_df)}"
    )

    print(
        "Test class distribution:",
        test_df["tangent_sign_numeric"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )


print(f"\nSaved splits to:")
print(OUTPUT_DIR.resolve())