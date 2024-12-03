# Reproducing Results for Keypoint Identification Method

This guide provides step-by-step instructions to reproduce the results of our Keypoint Identification method.

---

## Dataset Setup

The dataset is structured as follows:

```
data/
├── folds
    ├── split_labels_fold_0.csv
    ├── split_labels_fold_1.csv
    ├── ...

├── original
    ├── annotations
        ├── mask
        │   ├── pid_1.jpg
        │   ├── pid_2.jpg
        │   ├── ...

    ├── images
        ├── pid_1.jpg
        ├── pid_2.jpg
        ├── ...

    ├── muscle_segmentation
        ├── pid_1.jpg
        ├── pid_2.jpg
        ├── ...
```



---

## Running the Pipeline

Follow these steps to process the dataset and run the pipeline:

### 0. Set Up the Environment

You can create the environment by running this command:

```bash
conda create --name myenv
conda activate myenv
```

And install the requirements using conda and conda-forge:

```bash 
conda install -c conda-forge --file requirements.txt 
```

---

### 1. Data Preparation

#### a. Convert Image Annotations to Text
Run the following script to transform image annotations into text format:
```bash
python data_scripts/image_to_text_annotations.py
```

#### b. Annotate the Data
Run the annotation script and modify the directories as needed:
```bash
python data_scripts/annotate.py
```

#### c. Create YOLO-Compatible Dataset
Generate the YOLO dataset and YAML configuration:
```bash
python data_scripts/create_dataset.py
python data_scripts/create_yamls.py
```

After completing these steps, the dataset structure should look like the following:

```
data/
├── folds
    ├── split_labels_fold_0.csv
    ├── split_labels_fold_1.csv
    ├── ...
├── original
    ├── annotations
        ├── txt
            ├── pid_1.txt
            ├── pid_2.txt
            ├── ...
        ├── yolo
            ├── pid_1.txt
            ├── pid_2.txt
            ├── ...
        ├── mask
            ├── ...
    ├── images
        ├── ...
    ├── muscle_segmentation
        ├── ...
├── yolo_dataset
    ├── fold_i
        ├── images
            ├── train/
            ├── val/
        ├── labels
            ├── train/
            ├── val/
```

---

### 2. Training the Model

Train the model using the following command:
```bash
python train.py
```

---

### 3. Inference and Evaluation

Run the evaluation script to generate predictions and calculate scores:
```bash
python evaluate.py
```

This will output a folder for each fold containing:
- **Predictions**: Model-generated keypoints.
- **Scores CSV**: Evaluation metrics, including:
  - `pat`: Patient (image) ID.
  - `fold`: Fold the image belongs to.
  - `diff`: Number of pixel differences.
  - `score`: Percentage difference over the entire image.
  - `m_score`: Percentage difference over the muscle region.
  - `dice`: Dice score over the muscle region.

---

## Additional Notes

- Ensure that all paths are correctly set in the scripts before running them.
- The pipeline has been tested with the dataset structure and scripts provided above.

For further assistance, please contact the project team or refer to the code documentation.