# Reproducing Results for Keypoint Identification Method

This guide provides step-by-step instructions to reproduce the results of our Keypoint Identification method.

---

## Dataset Setup

The shared split files are created at repository level by `create_splits.py`. The keypoint data are structured as follows:

```text
data/
├── split_labels_fold_0.csv
├── split_labels_fold_1.csv
├── split_labels_fold_2.csv
├── split_labels_fold_3.csv
├── split_labels_fold_4.csv
│
└── keypoints/
    ├── original/
    │   ├── annotations/
    │   │   ├── mask/
    │   │   │   ├── pid_1.jpg
    │   │   │   └── ...
    │   │   ├── txt/
    │   │   │   ├── pid_1.txt
    │   │   │   └── ...
    │   │   └── yolo/
    │   │       ├── pid_1.txt
    │   │       └── ...
    │   ├── images/
    │   │   ├── pid_1.jpg
    │   │   └── ...
    │   └── muscle_segmentation/
    │       ├── pid_1.jpg
    │       └── ...
    │
    └── yolo_dataset_5fold/
```

The annotation folders contain:

- `mask/`: expert tangent-line annotations represented as images;
- `txt/`: two endpoint coordinates describing the expert tangent line;
- `yolo/`: YOLO pose labels containing one bounding box and three keypoints.

---

## Running the Pipeline

The commands below assume that they are run from the repository root.

### 0. Set Up the Environment

You can create the environment using the supplied requirements file:

```bash
conda create --name shoulderai-keypoints --file keypoints/requirements.txt
conda activate shoulderai-keypoints
```

### 1. Create the Shared Splits

Place the expert annotation and exclusion files under `data/`, then run:

```bash
python create_splits.py
```

This creates the five patient-grouped split files used by both the keypoint and segmentation approaches.

### 2. Data Preparation

#### a. Convert Image Annotations to Text

This one-time step is required only when the expert tangent-line endpoint files do not already exist:

```bash
python keypoints/data_scripts/image_to_text_annotations.py
```

The script converts each tangent-line mask in `annotations/mask/` into a two-line coordinate file under `annotations/txt/`.

#### b. Annotate the Data

This one-time manual step is required only when the YOLO pose labels do not already exist:

```bash
python keypoints/data_scripts/annotate.py
```

For each image, draw one bounding box and select three keypoints. Press:

- `s` to save a completed annotation;
- `r` to reset the current image;
- `q` to quit.

The resulting YOLO labels are saved under `annotations/yolo/`.

#### c. Create the YOLO-Compatible Dataset

Generate the five-fold YOLO train, validation, and test datasets and their YAML configurations:

```bash
python keypoints/data_scripts/create_dataset.py
```

The script:

- uses the shared `split_labels_fold_*.csv` files as the cohort definition;
- keeps repeated examinations from the same patient in the same data partition;
- uses fold `i` as test and fold `(i + 1) mod 5` as validation;
- includes existing `_aug` files only in training;
- excludes keypoint-only cases that are outside the study cohort;
- writes YAML files to `keypoints/yamls_5fold/`.

After completing this step, the generated dataset looks like:

```text
data/keypoints/yolo_dataset_5fold/
└── fold_i/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### 3. Training the Model

Place the pretrained pose checkpoint expected by `train.py` at:

```text
keypoints/model_yolo11l-pose.pt
```

Then train the five models:

```bash
python keypoints/train.py
```

The current configuration trains each fold for 150 epochs with an image size of 512 and automatic mixed precision disabled. Outputs are written to:

```text
keypoints/models_5fold/fold_i/
```

The best checkpoint for each fold is:

```text
keypoints/models_5fold/fold_i/weights/best.pt
```

### 4. Inference and Evaluation

Run the held-out test evaluation:

```bash
python keypoints/evaluate.py
```

This creates prediction overlays under:

```text
keypoints/evaluation/fold_i/pred/
```

and a combined result table at:

```text
keypoints/evaluation/results.csv
```

The result table includes:

- `pat`: patient/image ID;
- `fold`: held-out fold;
- `diff`: number of differing pixels;
- `score`: difference over the entire image;
- `m_score`: difference restricted to the supraspinatus mask;
- `dice`: Dice agreement within the supraspinatus mask.

---

## Additional Notes

- The split CSVs are the single source of truth for cohort membership.
- Validation and test sets contain only original images, never `_aug` images.
- Images without both an MRI file and a YOLO label are reported by the dataset-creation script.
- The final tangent-sign classification comparison is performed by `second_level_analysis.py` at repository level.

For further assistance, please contact the project team or refer to the code documentation.
