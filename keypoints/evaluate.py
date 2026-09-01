import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from utils import perform_pred
from datetime import datetime, timedelta
from pathlib import Path




def main():

    KEYPOINTS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = KEYPOINTS_DIR.parent
    DATA_DIR = REPO_ROOT / "data"

    RESULTS_BASE_PATH = KEYPOINTS_DIR / "evaluation"
    MODELS_BASE_PATH = KEYPOINTS_DIR / "models_5fold"
    DATA_BASE_PATH = DATA_DIR / "keypoints" / "yolo_dataset_5fold"
    GT_BASE_PATH = DATA_DIR / "keypoints" / "original" / "annotations" / "txt"
    MUSCLE_MASK_PATH = DATA_DIR / "keypoints" / "original" / "muscle_segmentation"
    

    folds = 5
    dfs = []
    for i in range(folds):
        work_dir = os.path.join(RESULTS_BASE_PATH,f"fold_{i}")
        model_path = os.path.join(
            MODELS_BASE_PATH,
            f"fold_{i}",
            "weights",
            "best.pt",
        )

        images_base_path = os.path.join(DATA_BASE_PATH,f"fold_{i}","images","test")
        
        images_out_base_path = os.path.join(work_dir,"pred")
        os.makedirs(images_out_base_path,exist_ok=True)
        scores = []

        val_patients = os.listdir(images_base_path)
        val_patients = [f.replace(".jpg","") for f in val_patients if f.endswith(".jpg")]
        gt_patients = os.listdir(GT_BASE_PATH)
        gt_patients = [f.replace(".txt","") for f in gt_patients if f.endswith(".txt")]
        muscle_mask_patients = os.listdir(MUSCLE_MASK_PATH)
        muscle_mask_patients = [f.replace(".jpg","") for f in muscle_mask_patients if f.endswith(".jpg")]
        
        available_patients = sorted(
            set(gt_patients)
            .intersection(set(val_patients))
            .intersection(set(muscle_mask_patients))
        )
        
        model = YOLO(model_path)

        for patient in tqdm(available_patients):
            image_path = os.path.join(images_base_path,f"{patient}.jpg")
            gt_path = os.path.join(GT_BASE_PATH,f"{patient}.txt")
            muscle_mask_path = os.path.join(MUSCLE_MASK_PATH,f"{patient}.jpg")
            patient_score = perform_pred(
                image_path,
                images_out_base_path,
                gt_path,
                muscle_mask_path,
                model,
            )
            scores.append(patient_score)
        score_path = os.path.join(work_dir,"results.csv")
        df = pd.DataFrame(scores)
        df = df[["pat","diff","score","m_score","dice"]]
        df["dice"] = df["dice"].fillna(0)
        df["fold"] = str(i)
        dfs.append(df)
    df_all = pd.concat(dfs)
    result_file = os.path.join(RESULTS_BASE_PATH,"results.csv")
    df_all.to_csv(result_file,index=False)



    print(f"Total rows: {len(df_all)}")
    print(f"Valid dice: {df_all['dice'].notna().sum()}")
    print(f"Valid m_score: {df_all['m_score'].notna().sum()}")

    print("\nYOLO surface difference:")
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        pct = 100 * (df_all["m_score"] < threshold).mean()
        print(f"<{threshold * 100:.0f}%: {pct:.1f}%")

    print("\nDice agreement:")
    for error in [0.05, 0.10, 0.15, 0.20]:
        threshold = 1 - error
        pct = 100 * (df_all["dice"] > threshold).mean()
        print(
            f"Dice error < {error * 100:.0f}% "
            f"(Dice > {threshold:.2f}): {pct:.1f}%"
        )

    print("\nOverall:")
    print(f"Dice: mean={df_all['dice'].mean():.4f}, std={df_all['dice'].std():.4f}")
    print(f"Raw pixel difference: mean={df_all['diff'].mean():.4f}, std={df_all['diff'].std():.4f}")
    print(f"Whole-image difference (%): mean={df_all['score'].mean() * 100:.2f}, std={df_all['score'].std() * 100:.2f}")
    print(f"Muscle-masked difference (%): mean={df_all['m_score'].mean() * 100:.2f}, std={df_all['m_score'].std() * 100:.2f}")

    print("\nPer fold:")
    summary = df_all.groupby("fold").agg({
        "dice": ["mean", "std", "count"],
        "diff": ["mean", "std"],
        "score": ["mean", "std"],
        "m_score": ["mean", "std"],
    })

    summary[("score", "mean")] *= 100
    summary[("score", "std")] *= 100
    summary[("m_score", "mean")] *= 100
    summary[("m_score", "std")] *= 100

    print(summary)



if __name__ == "__main__":
    main()