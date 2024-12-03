import cv2
import numpy as np
import os

def main():
    data_dir = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/original"

    mri_dir = os.path.join(data_dir,"images")
    annotations_dir = os.path.join(data_dir,"annotations","mask")
    outputs_dir = os.path.join(data_dir,"annotations","txt")

    mris = os.listdir(mri_dir)
    mris = [f for f in mris if f.endswith('.jpg')]
    annotations = os.listdir(annotations_dir)
    annotations = [f for f in annotations if f.endswith('.jpg')]
    patients = list(set(mris).intersection(set(annotations)))
    for patient in patients:
        mri_path = os.path.join(mri_dir, patient)
        annotation_path = os.path.join(annotations_dir, patient)
        
        mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        annotation_img = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        
        keypoints = get_keypoints(annotation_img)
        if len(keypoints) == 0:
            print(patient,"no lines")

        with open(os.path.join(outputs_dir, patient.replace(".jpg",".txt")),"w") as f:
            for keypoint in keypoints:
                keypoint = [str(x) for x in keypoint]
                f.write(f"{' '.join(keypoint)}\n")
def get_keypoints(img):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=80, minLineLength=2, maxLineGap=100)
    key_points = []
    if lines is None:
        return []
    key_points = []
    x1, y1, x2, y2 = lines[0][0]
    key_points.append((x1, y1))
    key_points.append((x2, y2))
    return key_points
if __name__ == '__main__':
    main()
