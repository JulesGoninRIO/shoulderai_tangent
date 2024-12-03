import pandas as pd
import os
import shutil
from tqdm import tqdm


def main():
    folds_path = "/data/soin/shoulder_ai/src/tangent sign/data" # Path of the csvs defining the folds. 
    data_src_path = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/original" # Path of the folder containing the original data
    data_dest_path = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/yolo_dataset" # Path of the destination folder
    
    folds_files = os.listdir(folds_path)
    folds_files = [f for f in folds_files if f.endswith(".csv")]

    for fold_file in tqdm(folds_files):
        df = pd.read_csv(os.path.join(folds_path,fold_file))
        train_patients = df[df["phase"] == "train"]["patient"].values.tolist()
        test_patients = df[df["phase"] == "test"]["patient"].values.tolist()
        fold_idx = fold_file.replace(".csv","")[-1]
        create_fold_directory(data_src_path,data_dest_path,train_patients,"train",fold_idx)
        create_fold_directory(data_src_path,data_dest_path,test_patients,"val",fold_idx)

def clear_dir(dire):
    if os.path.exists(dire):
        shutil.rmtree(dire)
    os.makedirs(dire)


def create_fold_directory(src_path,dest_path,patients,split,fold_idx):

    dest_images_folder = os.path.join(dest_path,f"fold_{fold_idx}","images",split)
    clear_dir(dest_images_folder)

    dest_annotations_folder = os.path.join(dest_path,f"fold_{fold_idx}","labels",split)
    clear_dir(dest_annotations_folder)

    patients += [f"{p}_aug" for p in patients]
    for patient in patients:
        src_image_path = os.path.join(src_path,"images",f"{patient}.jpg")
        src_annotation_path = os.path.join(src_path,"annotations","yolo", f"{patient}.txt")
        
        dest_image_path = os.path.join(dest_images_folder,f"{patient}.jpg")
        dest_annotation_path = os.path.join(dest_annotations_folder,f"{patient}.txt")
        
        if (not os.path.exists(src_image_path)) or (not os.path.exists(src_annotation_path)) :
            continue 
        shutil.copyfile(src_image_path,dest_image_path)
        shutil.copyfile(src_annotation_path,dest_annotation_path)


if __name__ == '__main__':
    main()