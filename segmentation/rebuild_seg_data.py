import yaml
from src.dataset import ROIDataModule
from pathlib import Path

SEG_DIR = Path(__file__).resolve().parent
REPO_ROOT = SEG_DIR.parent
CONFIG_PATH = SEG_DIR / "config" / "seg_tangent_sign.yml"

with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)

config["datamodule_args"]["root_dir"] = str(REPO_ROOT / "data")

dm = ROIDataModule(
    fold_index=0,
    **config["datamodule_args"],
)
print("Finished rebuilding segmentation data.")
