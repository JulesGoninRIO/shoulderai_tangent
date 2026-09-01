from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path


num_folds = 5

KEYPOINTS_DIR = Path(__file__).resolve().parent

original_model_path = KEYPOINTS_DIR / "model_yolo11l-pose.pt"
yamls_path = KEYPOINTS_DIR / "yamls_5fold"
models_path = KEYPOINTS_DIR / "models_5fold"

for i in range(num_folds):

    yaml_path = yamls_path / f"fold_{i}.yaml"

    model = YOLO(str(original_model_path))

    fold_start_time = datetime.now()

    results = model.train(
        data=yaml_path,
        project=models_path,
        name=f"fold_{i}",
        epochs=150,
        imgsz=512,
        amp=False,
        exist_ok=True,
    )

    fold_end_time = datetime.now()

    print(f"Training time: {fold_end_time - fold_start_time}")
    print(f"Saved to: {results.save_dir}")