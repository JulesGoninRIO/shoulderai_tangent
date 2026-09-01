from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from utils import load_ground_truth


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DATA_DIR = REPO_ROOT / "data"

DATA_BASE_PATH = (
    DATA_DIR / "keypoints" / "yolo_dataset_5fold"
)

GT_BASE_PATH = (
    DATA_DIR
    / "keypoints"
    / "original"
    / "annotations"
    / "txt"
)

MUSCLE_MASK_PATH = (
    DATA_DIR
    / "keypoints"
    / "original"
    / "muscle_segmentation"
)

OUT_DIR = (
    REPO_ROOT
    / "keypoints"
    / "evaluation"
    / "gt_all"
)

FOLDS = 10
ALPHA = 45
SHADE_RGB = (255, 255, 0)  # light yellow


def find_file(folder, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def add_muscle_shade(image, mask_path):
    mask = Image.open(mask_path).convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    alpha = mask.point(lambda x: ALPHA if x >= 128 else 0, "L")

    shade = Image.new("RGBA", image.size, (*SHADE_RGB, 0))
    shade.putalpha(alpha)

    return Image.alpha_composite(image.convert("RGBA"), shade).convert("RGB")


def main():
    for p in (DATA_BASE_PATH, GT_BASE_PATH, MUSCLE_MASK_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = set()
    skipped = 0

    for fold in range(FOLDS):
        images_dir = DATA_BASE_PATH / f"fold_{fold}" / "images" / "val"

        if not images_dir.exists():
            print(f"Missing: {images_dir}")
            continue

        image_files = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        for image_path in tqdm(image_files, desc=f"Fold {fold}"):
            patient = image_path.stem

            if patient in saved:
                continue

            gt_path = GT_BASE_PATH / f"{patient}.txt"
            mask_path = find_file(MUSCLE_MASK_PATH, patient)

            if not gt_path.exists() or mask_path is None:
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            image = add_muscle_shade(image, mask_path)

            gt_keypoints = load_ground_truth(gt_path)
            ImageDraw.Draw(image).line(gt_keypoints, fill="red", width=4)

            image.save(OUT_DIR / f"{patient}.jpg")
            saved.add(patient)

    print(f"Saved: {len(saved)}")
    print(f"Skipped missing GT/mask: {skipped}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()