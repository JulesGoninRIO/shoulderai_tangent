import os

import pandas as pd
import random

from utils.constants import COR_IMAGE_TYPES, SAG_IMAGE_TYPES, PHASES
from utils.utils_dicom import get_ROI_coord
from sklearn.model_selection import KFold


def save_split_labels(dicom_dir, root_dir, original_labels_path, test_perc=0.1):
    """Create and save .csv file containing train / test split of patients.

    Args:
        dicom_dir (str): Newly generated dicom directory path.
        root_dir (str): Internal root directory path, where to save the split labels.
        original_labels_path (str): Original labels path.
        test_perc (float, optional): Percentage of test data compared to
            train data (train + validation). Defaults to 0.1.
    """
    split_labels_path = os.path.join(root_dir, "split_labels.csv")
    if os.path.exists(split_labels_path):
        return

    # Define .csv columns
    columns = ["patient", "phase"] + SAG_IMAGE_TYPES + COR_IMAGE_TYPES
    df_split_labels = pd.DataFrame(columns=columns)

    df_original_labels = pd.read_csv(original_labels_path)
    

    for patient in os.listdir(dicom_dir):
        # original_patient_row = df_original_labels.loc[
        #     df_original_labels["Patient"] == patient
        # ]
        # if len(original_patient_row) > 0:
        #     # From original labels, extract eligible patients (see comments)
        #     # Verify it isn't RX studies.
        #     if (
        #         original_patient_row["Eligible"].values[0]
        #         and not original_patient_row["is_RX_2D"].values[0]
        #     ):
        images_list = os.listdir(os.path.join(dicom_dir, patient))

                # Write 1 for each image type if the patient has a study with this image type
        new_row = {img: 1 for img in images_list if img in columns}
        new_row["patient"] = patient
        df_split_labels = pd.concat(
                    [df_split_labels, pd.DataFrame([new_row])], ignore_index=True
        )

    # Create train / test split randomly
    df_split_labels = split_train_test(df_split_labels, test_perc)

    # Sort and save labels
    df_split_labels = df_split_labels.sort_values(by="patient")
    df_split_labels.reset_index(drop=True)
    df_split_labels.to_csv(split_labels_path)


def split_train_test(df_split_labels, test_perc):
    """Create and apply random train / test split, according to the proportions
    of patients with certain types of images, to ensure that for each image type,
    the test percentage is approximately the same.
    Attention: the logic must be adapted when new data arrives!

    Args:
        df_split_labels (DataFrame): DataFrame containing image types per patient.
            Where to write to train / test split.
        test_perc (float): Test data percentage compared to train data (train + validation).

    Returns:
        DataFrame: Updated df_split_labels with train / test split.
    """
    # TODO: When a patient has images for both shoulders, should both of them be
    # in the same train / test set? Is it a problem otherwise?
    # TODO change split logic when new data arrives:
    # Attention to keep same split as here but augment it with new data
    # Split logic:
    # T2_fs_cor and T2_fs_sag are almost present for all patients: can be ignored in logic
    # T1_in_phase_sag, T1_fat_sag and T1_water_sag have exactly the same patients
    # -> only consider T1_in_phase_sag
    # If T1_sag, T1_in_phase_sag, T1_fs_cor present: take test_perc of those patients.
    # Elif T1_sag present: also take test_per in any of those patients.
    # Else: don't consider (only 4 patients left so none will be taken as test)
    three_image_types_present_patients = []
    left_with_T1_sag_patients = []

    for i, row in df_split_labels.iterrows():
        if not (
            pd.isna(row["T1_sag"])
            or pd.isna(row["T1_in_phase_sag"])
            or pd.isna(row["T1_fs_cor"])
        ):  # 21 rows
            three_image_types_present_patients.append(row["patient"])
        elif row["T1_sag"]:  # 27 rows
            left_with_T1_sag_patients.append(row["patient"])

    # Compute number of test patients for each category
    num_test_three_images = round(test_perc * len(three_image_types_present_patients))
    num_test_left_T1_sag = round(test_perc * len(left_with_T1_sag_patients))

    # Perform split randomly on each category
    random.seed(22)
    test_pat_1 = random.sample(
        three_image_types_present_patients, num_test_three_images
    )
    random.seed(22)
    test_pat_2 = random.sample(left_with_T1_sag_patients, num_test_left_T1_sag)
    test_patients = test_pat_1 + test_pat_2

    df_split_labels["phase"] = df_split_labels["patient"].apply(
        lambda p: "test" if p in test_patients else "train"
    )

    # Print percentages per image
    for img_type in SAG_IMAGE_TYPES + COR_IMAGE_TYPES:
        test_perc = len(
            df_split_labels[img_type].loc[df_split_labels["phase"] == "test"]
        ) / len(df_split_labels[~df_split_labels[img_type].isnull()])
        print("Test percentage for image {}: {}".format(img_type, test_perc))
    return df_split_labels


def save_labels_ROI_slice_sag(original_labels_path, data_dir, split_labels_path):
    """Create and save updated Sagittal view ROI slice labels, with English translation.

    Args:
        original_labels_path (str): Original labels .csv file path.
        data_dir (str): Newly generated data directory path,
            where ROI labels output will be written.
        split_labels_path (str): Train / test labels path.
    """
    os.makedirs(data_dir, exist_ok=True)

    new_labels_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(new_labels_path):
        return

    df_labels = pd.read_csv(original_labels_path)
    df_split_labels = pd.read_csv(split_labels_path)

    # Define new ROI labels names
    df_ROI_slice_labels = pd.DataFrame(
        columns=[
            "patient",
            "tangent_sign",
            "fat_subscapular",
            "fat_supraspinatus",
            "fat_infraspinatus",
            "fat_teres_minor",
        ]
    )

    # For all eligible patients (defined in df_split_labels), translate label names
    eligible_patients = df_split_labels["patient"].tolist()
    for i, row in df_labels.iterrows():
        if row["Patient"] in eligible_patients:
            new_row = {}
            new_row["patient"] = row["Patient"]
            new_row["tangent_sign"] = row["Signe de la tangeante"]
            new_row["fat_subscapular"] = row["IG sous-scapulaire"]
            new_row["fat_supraspinatus"] = row["IG sus-épineux"]
            new_row["fat_infraspinatus"] = row["IG sous-épineux"]
            new_row["fat_teres_minor"] = row["IG pt rond"]
            df_ROI_slice_labels = pd.concat(
                [df_ROI_slice_labels, pd.DataFrame([new_row])], ignore_index=True
            )

    # Save new labels
    df_ROI_slice_labels = df_ROI_slice_labels.sort_values(by="patient")
    df_ROI_slice_labels.reset_index(drop=True)
    df_ROI_slice_labels.to_csv(new_labels_path)


def save_labels_ROI_slice_cor(original_labels_path, data_dir, split_labels_path):
    """Create and save updated Coronal view ROI slice labels, with English translation.

    Args:
        original_labels_path (str): Original labels .csv file path.
        data_dir (str): Newly generated data directory path,
            where ROI labels output will be written.
        split_labels_path (str): Train / test labels path.
    """
    os.makedirs(data_dir, exist_ok=True)

    new_labels_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(new_labels_path):
        return

    df_labels = pd.read_csv(original_labels_path)
    df_split_labels = pd.read_csv(split_labels_path)

    # Define new ROI labels names
    df_ROI_slice_labels = pd.DataFrame(
        columns=["patient", "cuff", "retraction_supraspinatus"]
    )

    # For all eligible patients (defined in df_split_labels), translate label names
    eligible_patients = df_split_labels["patients"].tolist()
    for i, row in df_labels.iterrows():
        if row["patient"] in eligible_patients:
            new_row = {}
            new_row["patient"] = row["Patient"]
            new_row["retraction_supraspinatus"] = row[
                "Rétraction sus-épineux selon Patte"
            ]  # TODO is this label useful? Check with doctors

            if row["Coiffe"] == "Non massive sous-scapulaire":
                cuff = "not_massive_subscapular"
            elif row["Coiffe"] == "Non massive sus-épineux":
                cuff = "not_massive_supraspinatus"
            elif row["Coiffe"] == "Non massive sous-épineux":
                cuff = "not_massive_infraspinatus"
            elif row["Coiffe"] == "Non massive petit rond":
                cuff = "not_massive_teres_minor"
            elif row["Coiffe"] == "Massive A":
                cuff = "massive_a"
            elif row["Coiffe"] == "Massive B":
                cuff = "massive_b"
            elif row["Coiffe"] == "Massive C":
                cuff = "massive_c"
            elif row["Coiffe"] == "Massive D":
                cuff = "massive_d"
            elif row["Coiffe"] == "Massive E":
                cuff = "massive_e"
            new_row["cuff"] = cuff

            df_ROI_slice_labels = pd.concat(
                [df_ROI_slice_labels, pd.DataFrame([new_row])], ignore_index=True
            )

    # Save new labels
    df_ROI_slice_labels = df_ROI_slice_labels.sort_values(by="patient")
    df_ROI_slice_labels.reset_index(drop=True)
    df_ROI_slice_labels.to_csv(new_labels_path)


def save_labels_all_slices(
    original_labels_path, data_dir, annot_dir, plane, split_labels_path
):
    """Create and save updated labels for all slices, with the ROI coordinate
    and pathological information for Sagittal images.
    Args:
        original_labels_path (str): Original labels .csv file path.
        data_dir (str): Newly generated data directory path,
            where all slices labels output will be written.
        annot_dir (str): Newly generated annotation directory path.
        plane (str): Plane of images and masks considered.
        split_labels_path (str): Train / test labels path.
    """
    os.makedirs(data_dir, exist_ok=True)

    new_labels_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(new_labels_path):
        return

    df_labels = pd.read_csv(original_labels_path)
    df_split_labels = pd.read_csv(split_labels_path)

    # Define new labels names for all slices
    columns = ["patient", "image_type", "ROI_coord", "num_slices"]
    if plane == "sagittal":
        columns.append("pathologic")
    df_all_slices_labels = pd.DataFrame(columns=columns)

    for phase in PHASES:
        images_dir = os.path.join(data_dir, phase)
        images = os.listdir(images_dir)

        for image in images:
            all_slices = os.listdir(os.path.join(images_dir, image))

            # Consider all eligible patients with the good image type in df_split_labels
            patients = df_split_labels.loc[
                (df_split_labels["phase"] == phase) & (df_split_labels[image] == 1),
                "patient",
            ].tolist()

            for patient in patients:
                annot_path = os.path.join(annot_dir, patient, image, image + ".csv")

                new_row = {}
                new_row["patient"] = patient
                new_row["image_type"] = image

                # Get ROI coordinate per patient
                ROI_coord = get_ROI_coord(annot_path, plane)
                if ROI_coord is not None:
                    new_row["ROI_coord"] = ROI_coord

                # Get number of slices per patient
                patient_slices = [x for x in all_slices if x.startswith(patient)]
                new_row["num_slices"] = len(patient_slices)

                # Add pathologic information depending on patient's Goutallier index, see paper
                # "Deep learning method for segmentation of rotator cuff muscles on MR images"
                if plane == "sagittal":
                    goutallier = df_labels.loc[
                        df_labels["Patient"] == patient, "IG sus-épineux"
                    ].values[0]
                    if goutallier > 1:
                        pathologic = 1
                    elif goutallier <= 1:  # Sometimes the row is empty
                        pathologic = 0
                    new_row["pathologic"] = pathologic

                # Verify if row is complete
                if len(new_row.keys()) == len(columns):
                    df_all_slices_labels = pd.concat(
                        [df_all_slices_labels, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )

    # Save new labels
    df_all_slices_labels = df_all_slices_labels.sort_values(
        by=["image_type", "patient"]
    )
    df_all_slices_labels.reset_index(drop=True)
    df_all_slices_labels.to_csv(new_labels_path)



def save_split_labels_crossval(dicom_dir, root_dir, n_splits=10):
    """Create and save .csv files containing k-fold cross-validation splits of patients.

    Args:
        dicom_dir (str): Newly generated dicom directory path.
        root_dir (str): Internal root directory path, where to save the split labels.
        original_labels_path (str): Original labels path.
        n_splits (int, optional): Number of folds for cross-validation. Defaults to 10.
    """
    # Define .csv columns
    split_labels_path = os.path.join(root_dir, "split_labels_fold_0.csv")
    if os.path.exists(split_labels_path):
        return

    # Define .csv columns
    columns = ["patient", "phase"] + SAG_IMAGE_TYPES + COR_IMAGE_TYPES
    df_split_labels = pd.DataFrame(columns=columns)
    for patient in os.listdir(dicom_dir):
        if patient in ["F0078", "F0072", "F0076", "F0079", "F0075", "F0080", "A0130", "F0003"]:
            continue
        images_list = os.listdir(os.path.join(dicom_dir, patient))
                # Write 1 for each image type if the patient has a study with this image type
        new_row = {img: 1 for img in images_list if img in columns}
        new_row["patient"] = patient
        df_split_labels = pd.concat(
                    [df_split_labels, pd.DataFrame([new_row])], ignore_index=True
                )


    # Create k-fold cross-validation splits
    df_split_labels = create_cross_validation_splits(df_split_labels, root_dir, n_splits)



def create_cross_validation_splits(df_split_labels, root_dir, n_splits):
    """Create k-fold cross-validation splits.

    Args:
        df_split_labels (DataFrame): DataFrame containing image types per patient.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        DataFrame: Updated df_split_labels with cross-validation splits.
    """
    # Define the categories for splitting
    three_image_types_present_patients = []
    left_with_T1_sag_patients = []

    for i, row in df_split_labels.iterrows():
        if not (
            pd.isna(row["T1_sag"])
            or pd.isna(row["T1_in_phase_sag"])
            or pd.isna(row["T1_fs_cor"])
        ):  # 21 rows
            three_image_types_present_patients.append(row["patient"])
        elif row["T1_sag"]:  # 27 rows
            left_with_T1_sag_patients.append(row["patient"])


    # Assign fold numbers and phases to patients with all three image types
    kf_three_images = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf_three_images.split(three_image_types_present_patients)):
        train_patients = [three_image_types_present_patients[i] for i in train_idx]
        val_patients = [three_image_types_present_patients[i] for i in val_idx]
        train_df = df_split_labels.loc[df_split_labels['patient'].isin(train_patients)].copy()
        train_df[['phase']] = "train"
        test_df = df_split_labels.loc[df_split_labels['patient'].isin(val_patients)].copy()
        test_df[['phase']] = "test"
        df_three = pd.concat([train_df, test_df]).reset_index(drop=True)
        fold_path = os.path.join(root_dir, f"split_labels_fold_{fold}.csv")
        df_three.to_csv(fold_path, index=False)


    # Assign fold numbers and phases to patients with only T1_sag
    kf_left_T1_sag = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf_left_T1_sag.split(left_with_T1_sag_patients)):
        train_patients = [left_with_T1_sag_patients[i] for i in train_idx]
        val_patients = [left_with_T1_sag_patients[i] for i in val_idx]
        train_df = df_split_labels.loc[df_split_labels['patient'].isin(train_patients)].copy()
        train_df[['phase']] = "train"
        test_df = df_split_labels.loc[df_split_labels['patient'].isin(val_patients)].copy()
        test_df[['phase']] = "test"
        df_sag = pd.concat([train_df, test_df]).reset_index(drop=True)
        fold_path = os.path.join(root_dir, f"split_labels_fold_{fold}.csv")
        df_sag.to_csv(fold_path, mode='a', index=False, header=False)
        print(f"Saved fold {fold} to {fold_path}")






