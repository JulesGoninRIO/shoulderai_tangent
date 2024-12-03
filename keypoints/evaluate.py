import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from utils import perform_pred



def main():
    RESULTS_BASE_PATH = "/data/soin/shoulder_ai/src/tangent sign/keypoints/evaluation" # path of where the images will be saved
    MODELS_BASE_PATH = "/data/soin/shoulder_ai/src/tangent sign/keypoints/models_" # path of the models, once trained
    DATA_BASE_PATH = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/yolo_dataset" # path of the training dataset
    GT_BASE_PATH = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/original/annotations/txt" # path of the annotations
    MUSCLE_MASK_PATH = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/original/muscle_segmentation" # path of the muscle segmentation
    
    folds = 10
    dfs = []
    for i in range(folds):
        work_dir = os.path.join(RESULTS_BASE_PATH,f"fold_{i}")
        model_path = os.path.join(MODELS_BASE_PATH,f"fold_{i}","weights","best.pt")

        images_base_path = os.path.join(DATA_BASE_PATH,f"fold_{i}","images","val") 
        
        images_out_base_path = os.path.join(work_dir,"pred")
        os.makedirs(images_out_base_path,exist_ok=True)



        scores = []

        val_patients = os.listdir(images_base_path)
        val_patients = [f.replace(".jpg","") for f in val_patients if f.endswith(".jpg")]
        gt_patients = os.listdir(GT_BASE_PATH)
        gt_patients = [f.replace(".txt","") for f in gt_patients if f.endswith(".txt")]
        muscle_mask_patients = os.listdir(MUSCLE_MASK_PATH)
        muscle_mask_patients = [f.replace(".jpg","") for f in muscle_mask_patients if f.endswith(".jpg")]
        
        available_patients = list(set(gt_patients).intersection(set(val_patients)).intersection(set(muscle_mask_patients)))
        
        model = YOLO(model_path)

        for patient in tqdm(available_patients):
            image_path = os.path.join(images_base_path,f"{patient}.jpg")
            gt_path = os.path.join(GT_BASE_PATH,f"{patient}.txt")
            muscle_mask_path = os.path.join(MUSCLE_MASK_PATH,f"{patient}.jpg")
            patient_score = perform_pred(image_path,images_out_base_path,gt_path,muscle_mask_path,model)
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



if __name__ == "__main__":
    main()