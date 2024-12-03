from ultralytics import YOLO
import os

num_folds = 10

original_model_path = "model_yolo11l-pose.pt" # path of the pretrained model, can be found here (https://docs.ultralytics.com/tasks/pose/)
yamls_path = "yamls"

for i in range(num_folds):
    model = YOLO(original_model_path)
    yaml_path = os.path.join(yamls_path,f"fold_{i}.yaml")
    results = model.train(data=yaml_path,project="models",name=f"fold_{i}", epochs=100, imgsz=512)