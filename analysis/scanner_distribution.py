from pathlib import Path
import pandas as pd
import pydicom


DICOM_ROOT = Path(
    r"/data/soin/shoulder_ai/data/old/DATA/dicom/new_dicom_2024/"
)

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent

SPLIT_DIR = REPO_ROOT / "data"

OUT_CSV = ANALYSIS_DIR / "MRI_machine_by_scan.csv"
OUT_PNG = ANALYSIS_DIR / "MRI_machine_distribution.png"

PID_COL = "patient"   


# ------------------------------------------------------------
# Patients actually used in the ML splits
# ------------------------------------------------------------

patients = set()

for csv_file in SPLIT_DIR.glob("*0.csv"):
    df = pd.read_csv(csv_file)
    patients.update(df[PID_COL].astype(str))

print(f"Patients in splits: {len(patients)}")


# ------------------------------------------------------------
# Read one DICOM from each scan folder
# ------------------------------------------------------------

rows = []

for pid in sorted(patients):

    patient_dir = DICOM_ROOT / pid

    if not patient_dir.exists():
        print(f"Missing patient folder: {pid}")
        continue

    # folders containing actual DICOM files
    scan_dirs = sorted({
        p.parent
        for p in patient_dir.rglob("*")
        if p.is_file()
    })

    for scan_dir in scan_dirs:

        for file in scan_dir.iterdir():
            if not file.is_file():
                continue

            try:
                ds = pydicom.dcmread(
                    file,
                    stop_before_pixels=True
                )

                rows.append({
                    "pid": pid,
                    "scan_path": str(scan_dir),
                    "manufacturer": str(
                        getattr(ds, "Manufacturer", "")
                    ),
                    "model": str(
                        getattr(ds, "ManufacturerModelName", "")
                    ),
                    "field_strength": str(
                        getattr(ds, "MagneticFieldStrength", "")
                    ),
                    "serial_number": str(
                        getattr(ds, "DeviceSerialNumber", "")
                    ),
                    "station_name": str(
                        getattr(ds, "StationName", "")
                    ),
                })

                break

            except Exception:
                continue


scans = pd.DataFrame(rows)


# ------------------------------------------------------------
# Define machine
# ------------------------------------------------------------

machine_cols = [
    "manufacturer",
    "model",
    "field_strength",
    "serial_number",
]

scans["machine"] = (
    scans[machine_cols]
    .fillna("")
    .astype(str)
    .agg(" | ".join, axis=1)
)


# ------------------------------------------------------------
# Check patients scanned on >1 machine
# ------------------------------------------------------------

machines_per_patient = (
    scans.groupby("pid")["machine"]
    .nunique()
)

mixed = machines_per_patient[
    machines_per_patient > 1
]

print("\nPatients scanned on >1 machine:")
print(mixed)

if len(mixed):
    print(
        scans[
            scans["pid"].isin(mixed.index)
        ][
            ["pid", "scan_path", "machine", "station_name"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

# Patient-level distribution
patient_machine = (
    scans.groupby("pid")["machine"]
    .first()
)

print("\nMRI machine distribution by patient:")
print(patient_machine.value_counts())

print("\nPercent:")
print(
    patient_machine.value_counts(normalize=True)
    .mul(100)
    .round(1)
)


# Save details
scans.to_csv(OUT_CSV, index=False)

# ------------------------------------------------------------
# Plot MRI machine distribution
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

patient_model = (
    scans.groupby("pid")["model"]
    .first()
)

counts = patient_model.value_counts()

colors = plt.cm.viridis(
    np.linspace(0.15, 0.9, len(counts))
)

plt.figure(figsize=(10, 5))

plt.bar(
    counts.index,
    counts.values,
    color=colors,
    edgecolor="black",
)

plt.xlabel("MRI Machine Model")
plt.ylabel("Number of Patients")
plt.title("MRI Machine Distribution in the Study Cohort")

plt.xticks(
    rotation=45,
    ha="right",
)

# Horizontal grid lines every 5 patients
ymax = int(np.ceil(counts.max() / 5) * 5)
plt.yticks(np.arange(0, ymax + 1, 5))
plt.ylim(0, ymax + 1)

plt.grid(
    axis="y",
    linestyle="-",
    alpha=0.3,
)

# Keep grid behind the bars
plt.gca().set_axisbelow(True)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)

plt.show()