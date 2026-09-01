from pathlib import Path
import pandas as pd
import shutil
import re


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
KEYPOINTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = KEYPOINTS_DIR.parent
DATA_DIR = REPO_ROOT / "data"

SPLIT_DIR = DATA_DIR

KP_ORIG = DATA_DIR / "keypoints" / "original"
IMG_DIR = KP_ORIG / "images"
YOLO_LABEL_DIR = KP_ORIG / "annotations" / "yolo"

OUT_DIR = DATA_DIR / "keypoints" / "yolo_dataset_5fold"
YAML_DIR = KEYPOINTS_DIR / "yamls_5fold"


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
N_FOLDS = 5

USE_AUG_IN_TRAIN = True
USE_AUG_IN_VAL = False

WRITE_YAMLS = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def strip_aug(case_id: str) -> str:
    return re.sub(r"_aug$", "", str(case_id))


def base_patient_id(case_id: str) -> str:
    """
    Group repeated examinations from the same patient.

    Examples
    --------
    F0051     -> F0051
    F0051-2   -> F0051
    F0051_aug -> F0051
    """
    case_id = strip_aug(case_id)
    return re.sub(r"-\d+$", "", case_id)


def has_keypoint_pair(case_id: str) -> bool:
    """Return True when both the MRI JPG and YOLO label exist."""
    return (
        (IMG_DIR / f"{case_id}.jpg").exists()
        and (YOLO_LABEL_DIR / f"{case_id}.txt").exists()
    )


def add_existing_aug_versions(cases, use_aug: bool):
    """
    Return the existing image/label pairs for a set of original cases.

    Augmented versions are included only when requested.
    """
    files = set()

    for case_id in cases:
        if has_keypoint_pair(case_id):
            files.add(case_id)

        if use_aug:
            aug_id = f"{case_id}_aug"
            if has_keypoint_pair(aug_id):
                files.add(aug_id)

    return sorted(files)


def copy_cases(case_ids, fold_dir: Path, split: str):
    """Copy images and YOLO labels into one fold/split directory."""
    img_out = fold_dir / "images" / split
    lab_out = fold_dir / "labels" / split

    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        shutil.copy2(
            IMG_DIR / f"{case_id}.jpg",
            img_out / f"{case_id}.jpg",
        )
        shutil.copy2(
            YOLO_LABEL_DIR / f"{case_id}.txt",
            lab_out / f"{case_id}.txt",
        )


def load_split_files():
    """
    Load all fold CSVs and verify that they describe the same study cohort.

    The split CSVs are the single source of truth for cohort inclusion.
    Exclusions must therefore be applied upstream in create_splits.py.
    """
    split_dfs = {}

    for fold in range(N_FOLDS):
        split_path = SPLIT_DIR / f"split_labels_fold_{fold}.csv"

        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {split_path}"
            )

        df = pd.read_csv(split_path)

        required = {"patient", "phase"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{split_path.name} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        df = df.copy()
        df["patient"] = df["patient"].astype(str).str.strip()
        df["phase"] = df["phase"].astype(str).str.strip().str.lower()

        invalid_phases = sorted(
            set(df["phase"]) - {"train", "test"}
        )

        if invalid_phases:
            raise ValueError(
                f"{split_path.name} contains invalid phases: "
                + ", ".join(invalid_phases)
            )

        if df["patient"].duplicated().any():
            duplicates = sorted(
                df.loc[df["patient"].duplicated(), "patient"].unique()
            )
            raise ValueError(
                f"{split_path.name} contains duplicate patients: "
                + ", ".join(duplicates)
            )

        split_dfs[fold] = df

    reference_cases = set(split_dfs[0]["patient"])

    for fold in range(1, N_FOLDS):
        fold_cases = set(split_dfs[fold]["patient"])

        if fold_cases != reference_cases:
            missing_from_fold = sorted(reference_cases - fold_cases)
            extra_in_fold = sorted(fold_cases - reference_cases)

            raise ValueError(
                f"Fold {fold} does not contain the same study cohort as fold 0.\n"
                f"Missing from fold {fold}: {missing_from_fold}\n"
                f"Extra in fold {fold}: {extra_in_fold}"
            )

    return split_dfs, reference_cases


def assert_patient_disjoint(train_cases, val_cases, test_cases, fold):
    """
    Verify that no base patient appears in more than one split.

    This protects against leakage when repeated examinations such as
    F0051 and F0051-2 are present.
    """
    train_base = {base_patient_id(c) for c in train_cases}
    val_base = {base_patient_id(c) for c in val_cases}
    test_base = {base_patient_id(c) for c in test_cases}

    overlaps = {
        "train/val": train_base & val_base,
        "train/test": train_base & test_base,
        "val/test": val_base & test_base,
    }

    overlaps = {
        name: sorted(values)
        for name, values in overlaps.items()
        if values
    }

    if overlaps:
        raise ValueError(
            f"Base-patient leakage detected in fold {fold}: {overlaps}"
        )


# ------------------------------------------------------------
# Load authoritative study cohort
# ------------------------------------------------------------
split_dfs, study_cases = load_split_files()

print(f"Study cohort defined by split CSVs: {len(study_cases)} cases")


# ------------------------------------------------------------
# Determine which study cases have usable keypoint annotations
# ------------------------------------------------------------
keypoint_original_cases = {
    p.stem
    for p in IMG_DIR.glob("*.jpg")
    if not p.stem.endswith("_aug") and has_keypoint_pair(p.stem)
}

available_study_cases = study_cases & keypoint_original_cases
missing_keypoint_cases = sorted(study_cases - keypoint_original_cases)
extra_keypoint_only_cases = sorted(keypoint_original_cases - study_cases)

print(f"Original cases with keypoint image + label: {len(keypoint_original_cases)}")
print(f"Study cases with keypoint image + label:    {len(available_study_cases)}")
print(f"Study cases missing keypoint pair:          {len(missing_keypoint_cases)}")
print(f"Keypoint-only cases outside study cohort:   {len(extra_keypoint_only_cases)}")

for case_id in missing_keypoint_cases:
    print(f"  missing keypoint pair: {case_id}")

if extra_keypoint_only_cases:
    print(
        "\nKeypoint-only cases outside the study cohort are NOT used for "
        "training, validation, or testing:"
    )
    for case_id in extra_keypoint_only_cases:
        print(f"  outside cohort: {case_id}")


# ------------------------------------------------------------
# Recreate YOLO dataset
# ------------------------------------------------------------
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for fold in range(N_FOLDS):

    # Current fold = TEST
    df = split_dfs[fold]

    test_cases_all = set(
        df.loc[df["phase"] == "test", "patient"]
    )

    # Next fold = VALIDATION
    val_fold = (fold + 1) % N_FOLDS
    val_df = split_dfs[val_fold]

    val_cases_all = set(
        val_df.loc[val_df["phase"] == "test", "patient"]
    )

    # Remaining cases = TRAIN
    train_cases_all = (
        study_cases
        - test_cases_all
        - val_cases_all
    )

    # Keep only cases that actually have an image + YOLO label.
    train_cases = train_cases_all & keypoint_original_cases
    val_cases = val_cases_all & keypoint_original_cases
    test_cases = test_cases_all & keypoint_original_cases

    # Repeated examinations from the same base patient must never
    # cross train/validation/test boundaries.
    assert_patient_disjoint(
        train_cases,
        val_cases,
        test_cases,
        fold,
    )

    train_files = add_existing_aug_versions(
        train_cases,
        USE_AUG_IN_TRAIN,
    )
    val_files = add_existing_aug_versions(
        val_cases,
        USE_AUG_IN_VAL,
    )
    test_files = add_existing_aug_versions(
        test_cases,
        False,
    )

    # Validation/test must never contain pre-generated augmentations.
    if any(case_id.endswith("_aug") for case_id in val_files):
        raise ValueError(
            f"Augmented validation case detected in fold {fold}."
        )

    if any(case_id.endswith("_aug") for case_id in test_files):
        raise ValueError(
            f"Augmented test case detected in fold {fold}."
        )

    fold_dir = OUT_DIR / f"fold_{fold}"

    copy_cases(train_files, fold_dir, "train")
    copy_cases(val_files, fold_dir, "val")
    copy_cases(test_files, fold_dir, "test")

    print(
        f"\nfold {fold}: "
        f"train={len(train_files)} files from {len(train_cases)} original cases, "
        f"val={len(val_files)} files from {len(val_cases)} original cases, "
        f"test={len(test_files)} files from {len(test_cases)} original cases"
    )

    missing_train = sorted(train_cases_all - keypoint_original_cases)
    missing_val = sorted(val_cases_all - keypoint_original_cases)
    missing_test = sorted(test_cases_all - keypoint_original_cases)

    if missing_train:
        print(f"  train cases unavailable for YOLO: {missing_train}")
    if missing_val:
        print(f"  val cases unavailable for YOLO:   {missing_val}")
    if missing_test:
        print(f"  test cases unavailable for YOLO:  {missing_test}")


# ------------------------------------------------------------
# Write YOLO YAMLs
# ------------------------------------------------------------
if WRITE_YAMLS:
    YAML_DIR.mkdir(parents=True, exist_ok=True)

    for fold in range(N_FOLDS):
        fold_path = (OUT_DIR / f"fold_{fold}").resolve()
        yaml_path = YAML_DIR / f"fold_{fold}.yaml"

        # Use an absolute path so Ultralytics does not depend on the
        # current working directory from which train.py is launched.
        yaml_text = f"""path: "{fold_path.as_posix()}"
train: images/train
val: images/val
test: images/test
kpt_shape: [3, 2]
names:
  0: scapula
"""

        yaml_path.write_text(
            yaml_text,
            encoding="utf-8",
        )

    print(f"\nWrote YAMLs to: {YAML_DIR.resolve()}")

print(f"\nYOLO dataset written to: {OUT_DIR.resolve()}")
print("\nDone.")
