# Reproducing Results for U-Net Segmentation Method

This guide provides step-by-step instructions to reproduce the results of the U-Net tangent-line segmentation method.

---

## Local Environment

You can create and activate an environment:

```bash
conda create --name shoulderai-segmentation python
conda activate shoulderai-segmentation
```

Install the supplied requirements:

```bash
python -m pip install -r segmentation/requirements.txt
```

---

## Dataset Setup

The shared study cohort and five-fold splits are created at repository level:

```bash
python create_splits.py
```

The segmentation data are stored under:

```text
data/
├── split_labels_fold_0.csv
├── split_labels_fold_1.csv
├── split_labels_fold_2.csv
├── split_labels_fold_3.csv
├── split_labels_fold_4.csv
│
└── sagittal/
    ├── images/
    │   └── T1_sag/
    │       ├── pid_1.jpg
    │       ├── pid_1_aug.jpg
    │       └── ...
    └── masks/
        └── T1_sag/
            ├── tangent_sign/
            │   ├── pid_1.jpg
            │   ├── pid_1_aug.jpg
            │   └── ...
            └── supraspinatus/
                ├── pid_1.jpg
                ├── pid_1_aug.jpg
                └── ...
```

Corresponding images and masks must use the same filename.

For fold `i`:

- fold `i` is the test set;
- fold `(i + 1) mod 5` is the validation set;
- the remaining folds form the training set.

Augmented files are included only in training.

---

## Conversion of Original Data to JPG Format

If the JPG images and masks are already available in the structure above, this step can be skipped.

To recreate them from the original DICOM images and annotation files, first configure the external raw-data location used by the segmentation utilities, then run:

```bash
python segmentation/rebuild_seg_data.py
```

This prepares the local `data/sagittal/` image and mask folders. The shared split CSVs must already exist before this step.

---

## Training the Tangent-Sign U-Net

The model and data settings are stored in:

```text
segmentation/config/seg_model.yml
segmentation/config/seg_tangent_sign.yml
```

To train the five segmentation models, run:

```bash
python segmentation/segmentation_model.py --mode train
```

The current configuration uses:

- a U-Net architecture;
- an EfficientNet-B7 encoder initialized with ImageNet weights;
- five-fold patient-grouped cross-validation;
- early stopping with a patience of 10 validation epochs;
- mixed-precision training;
- random augmentation only in the training pipeline.

The best checkpoint from each fold is saved under:

```text
segmentation/check/
```

Offline Weights & Biases logs are written under:

```text
segmentation/wandb/
```

---

## Testing the Tangent-Sign U-Net

To evaluate the five saved checkpoints, run:

```bash
python segmentation/segmentation_model.py --mode test
```

During testing, the pipeline:

- loads one checkpoint for each fold;
- calibrates the pixel-probability threshold on the validation fold;
- evaluates the held-out test fold;
- uses deterministic validation/test preprocessing without random CLAHE;
- runs test inference in full precision;
- converts each predicted mask into a fitted tangent line;
- evaluates agreement within the supraspinatus mask;
- saves per-case overlays and result CSV files.

The generated outputs include:

- one `*_losses.csv` file per evaluated case;
- `merged_numbers.csv`;
- fold-specific overlay images;
- pooled Dice and surface-difference summaries printed to the terminal.

The result directory is prefixed with the mean test loss after evaluation.

---

## Final Tangent-Sign Classification

The segmentation localization results are one part of the final paper analysis. After both the U-Net and YOLO models have been trained, run:

```bash
python second_level_analysis.py
```

This calculates the percentage of supraspinatus muscle above the predicted line and evaluates tangent-sign classification using the three threshold-selection strategies.

---

## Additional Notes

- Delete `second_level_results/continuous_scores.csv` before rerunning second-level inference after changing a model, checkpoint, preprocessing rule, or cohort definition.
