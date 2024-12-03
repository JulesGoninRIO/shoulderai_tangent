import os
import yaml
from tqdm import tqdm

data_path = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/yolo_dataset" # path of the yolo_dataset
dest = "yamls" # path of where the yaml files will be created


fold_dirs = [d for d in os.listdir(data_path) if d.startswith("fold_") and os.path.isdir(os.path.join(data_path, d))]
    

if not os.path.exists(dest):
    os.makedirs(dest)

for fold_dir in tqdm(fold_dirs):
    fold_path = os.path.join(data_path, fold_dir)
    yaml_content = {
        "path": fold_path,
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [3, 2],
        "names": {
            0: "scapula"
        }
    }

    yaml_file_path = os.path.join(dest, f"{fold_dir}.yaml")
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)