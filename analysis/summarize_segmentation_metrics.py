from pathlib import Path
import re
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent

RESULTS_RUN = "0.9912_2026-08-16_results"

results_dir = (
    REPO_ROOT
    / "segmentation"
    / RESULTS_RUN
)

split_dir = REPO_ROOT / "data"

if not results_dir.exists():
    raise SystemExit(f"Path does not exist: {results_dir}")

def clean_case_id(x):
    x = str(x).strip()
    x = Path(x).stem
    x = re.sub(r"_aug$", "", x)
    return x

# Build patient -> test fold map
case_to_fold = {}

for fold in range(5):
    split_file = split_dir / f"split_labels_fold_{fold}.csv"
    df = pd.read_csv(split_file)

    test_cases = df.loc[df["phase"] == "test", "patient"].astype(str)

    for case in test_cases:
        case_to_fold[clean_case_id(case)] = fold


rows = []

for f in sorted(results_dir.rglob("*_losses.csv")):
    try:
        vals = pd.read_csv(f, header=None).iloc[:, 0].dropna().tolist()

        patient = clean_case_id(vals[0])


        dice = float(vals[1])
        surface_difference = float(vals[2])

        fold = case_to_fold.get(patient)

        if fold is None:
            print(
                "Skipping result outside the "
                f"current study cohort: {patient}"
            )
            continue

        rows.append({
            "fold": fold,
            "patient": patient,
            "dice": dice,
            "surface_difference_percent": surface_difference,
            "source_file": str(f),
        })

    except Exception as e:
        print(f"Could not read {f}: {e}")

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No usable *_losses.csv files found.")

missing = df[df["fold"].isna()]
if not missing.empty:
    print("\nWarning: could not infer fold for these patients:")
    print(missing[["patient", "source_file"]].to_string(index=False))

df = df.dropna(subset=["fold"]).copy()
df["fold"] = df["fold"].astype(int)

summary = df.groupby("fold").agg(
    n=("patient", "count"),
    dice_mean=("dice", "mean"),
    dice_std=("dice", "std"),
    surface_difference_mean=("surface_difference_percent", "mean"),
    surface_difference_std=("surface_difference_percent", "std"),
).reset_index()

overall = pd.DataFrame([{
    "fold": "overall",
    "n": len(df),
    "dice_mean": df["dice"].mean(),
    "dice_std": df["dice"].std(),
    "surface_difference_mean": df["surface_difference_percent"].mean(),
    "surface_difference_std": df["surface_difference_percent"].std(),
}])

summary = pd.concat([summary, overall], ignore_index=True)

print("\nSegmentation Dice and surface difference:")
print(summary.to_string(index=False))

out_all = results_dir / "segmentation_collected_dice_surface.csv"
out_summary = results_dir / "segmentation_foldwise_dice_surface_summary.csv"

df.to_csv(out_all, index=False)
summary.to_csv(out_summary, index=False)

print(f"\nSaved collected data: {out_all}")
print(f"Saved summary:        {out_summary}")