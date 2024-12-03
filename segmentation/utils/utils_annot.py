import distutils.dir_util as dir_util
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.annotations_model import*
from utils.constants import SHAPE
from utils.utils_dicom import get_new_image_name_and_dir, check_and_create_dir, save_ROI_jpg, dimension_dicom

# Remove these annotations because they cannot be a part of the pipeline:
# They are made on a different sequence than other annotations.
annotations_to_remove = {"F0007": "T1_in_phase_sag", "F0039": "T1_fat_sag"}


def is_circle(X, Y, cutoff=0.99):
    """Function that verifies whether a polygon is a circle.

    Args:
        X (ndarray): X coordinates.
        Y (ndarray): Y coordinates.
        cutoff (float, optional): Cutoff parameter. Defaults to 0.99.

    Returns:
        boolean: True if polygon is a circle, False otherwise.
    """
    if X.shape[0] < 3:
        res = False
    else:
        pgon = Polygon(zip(X, Y))
        circ = 4 * np.pi * pgon.area / pgon.length / pgon.length
        res = circ > cutoff
    return res


def create_mask(points, shape, new_shape):
    """Helper function to generate binary mask from points.

    Args:
        points (ndarray): Array containing points delimiting the shape to draw on the mask.
        shape (tuple): Tuple containing X-axis and Y-axis dimensions of the mask.

    Returns:
        ndarray: Binary mask.
    """
    scale_x = new_shape[1] / shape[1]
    scale_y = new_shape[0] / shape[0]
    scale_points = np.array([[int(x*scale_x), int(y*scale_y)] for x, y in points[0]])

    mask = np.zeros(new_shape, dtype="uint8")
    cv2.polylines(mask, [scale_points], isClosed=True, color=255)
    cv2.fillPoly(mask, [scale_points], 255)
    return mask


def create_new_annot_dir(original_dir, new_dir):
    """Create from original annotations directory a new directory with more descriptive file names.

    Args:
        original_dir (str): Original annotation directory path.
        new_dir (str): New annotation directory path.
    """

    exists = check_and_create_dir(new_dir)
    if exists:
        return

    for p in os.listdir(original_dir):
        original_patient_dir = os.path.join(original_dir, p)
        list_studies = os.listdir(original_patient_dir)
        patient = p[-5:]

        # Test if patient has two studies (directory contains directories)
        double_study = False
        for s in list_studies:
            study_path = os.path.join(original_patient_dir, s)
            for x in os.listdir(study_path):
                if os.path.isdir(os.path.join(study_path, x)):
                    double_study = True

        # If patient has two studies, generate two folders: {patient_code}-1, {patient_code}-2
        if double_study:
            for idx, sub_patient in enumerate(list_studies):
                original_sub_patient_dir = os.path.join(
                    original_patient_dir, sub_patient
                )
                patient_numbered = patient + "-" + str(idx + 1) if idx > 0 else patient
                copy_annot_patient(original_sub_patient_dir, new_dir, patient_numbered)
        else:
            copy_annot_patient(original_patient_dir, new_dir, patient)


def copy_annot_patient(original_patient_dir, new_dir, patient):
    """Copy original annotations of a patient to new directory, with new image name convention.

    Args:
        original_patient_dir (str): Original patient annotation directory path.
        new_dir (str): New annotation directory path.
        patient (str): Patient id.
    """
    new_dir = os.path.join(new_dir, patient)

    for image_name in os.listdir(original_patient_dir):
        tmp_output_dir = new_dir
        image_path = os.path.join(original_patient_dir, image_name)
        image_name = image_name.lower()
        new_image_name, new_output_dir = get_new_image_name_and_dir(
            image_name, tmp_output_dir
        )

        # Remove unwanted annotations
        if annotations_to_remove.get(patient) == new_image_name:
            continue

        # Copy annotations
        new_image_path = os.path.join(new_output_dir, new_image_name)
        dir_util.copy_tree(image_path, new_image_path)

        # Rename files in patient annotation folder with new image name
        for f in os.listdir(new_image_path):
            old_f_path = os.path.join(new_image_path, f)
            if os.path.isfile(old_f_path):
                _, ext = os.path.splitext(f)
                new_f_path = os.path.join(new_image_path, new_image_name + ext)
                os.rename(old_f_path, new_f_path)


def save_annot_to_mask(annot_dir, data_dir, plane, img_types, split_labels_path, mask, dicom_dir):
    """
    From the .csv annotations, create the binary mask image for semantic segmentation.

    Args:
        annot_dir (str): Newly generated annotation directory path.
        data_dir (str): Newly generated data directory path, where mask output will be written.
        plane (str): Plane of images and masks considered. Either "sagittal" or "coronal".
        img_types (List): Image types considered per plane:
            Sagittal: ['T1_sag', 'T1_in_phase_sag', 'T1_fat_sag'], Coronal: ['T2_fs_cor'].
        split_labels_path (str): Train/test labels path.
        mask (str): Mask type or identifier.
        dicom_dir (str): DICOM directory where original images are stored.
    """
    masks_dir = os.path.join(data_dir, "masks")  # Unified masks directory
    if os.path.exists(os.path.relpath(masks_dir)):
        return
    os.makedirs(masks_dir, exist_ok=True)

    # Select patients that are listed in the split_labels
    df_split_labels = pd.read_csv(split_labels_path)
    patients = df_split_labels["patient"].values

    # Filter patients that actually have annotations
    patients = list(set(patients) & set(os.listdir(annot_dir)))

    # Loop through patients to retrieve and save masks
    for p in patients:
        patient_dir = os.path.join(annot_dir, p)
        selected_img_types = list(set(os.listdir(patient_dir)) & set(img_types))

        # Saving masks for sagittal or coronal planes
        if plane == "sagittal":
            if "T1_sag" in selected_img_types:
                save_masks_sag(
                    ["T1_sag"], patient_dir, p, os.path.join(masks_dir, "T1_sag"), mask, dicom_dir
                )
                # other_img_types = list(set(selected_img_types) - set(["T1_sag"]))
                # p = p + "_2"
                # save_masks_sag(
                #     other_img_types,
                #     patient_dir,
                #     p,
                #     os.path.join(masks_dir, "T1_sag"),
                #     mask,
                #     dicom_dir,
                # )
            else:
                save_masks_sag(
                    selected_img_types,
                    patient_dir,
                    p,
                    os.path.join(masks_dir, "other_T1_sag"),
                    mask,
                    dicom_dir,
                )
        else:
            save_masks_cor(selected_img_types, patient_dir, masks_dir)



def save_annot_to_mask_all(annot_dir, mask_dir,patient):
    """From the .csv annotations, create the binary mask image for any shape found.
    Function used to see which type of information was originally annotated.

    Args:
        annot_dir (str): Newly generated annotation directory path.
        mask_dir (str): Output mask directory path.
    """
    exists = check_and_create_dir(mask_dir)
    if exists:
        return

    for p in os.listdir(annot_dir):
        patient_dir = os.path.join(annot_dir, p)
        masks = defaultdict(list)
        for img_type in os.listdir(patient_dir):
            if img_type != "other":
                # Get annotations
                annot_path = os.path.join(patient_dir, img_type, img_type + ".csv")
                if patient.startswith("A") and patient !="A0103" or patient =="F0028" or patient == "A0098" or (patient == "F0015" and img_type=="T1_sag"):
                    annotations = NewOsirixAnnotationList()
                else:
                    annotations = OsirixAnnotationList()
                annotations.load_from_csv(annot_path)
                for annotation in annotations.annotations:
                    X, Y = np.array(annotation.px_X), np.array(annotation.px_Y)
                    annotation_points = np.dstack((X, Y)).astype(int)

                    # Create mask
                    mask = create_mask(annotation_points, SHAPE)
                    if len(X) == 1:
                        masks["point"].append(mask)
                    if len(X) == 2:
                        masks["line"].append(mask)
                    if len(X) > 2:
                        masks["polygon"].append(mask)

            for key, value in masks.items():
                tmp_mask_dir = os.path.join(mask_dir, p, key)
                os.makedirs(tmp_mask_dir, exist_ok=True)
                cv2.imwrite(os.path.join(tmp_mask_dir, img_type + ".jpg"), value)


def save_masks_sag(img_types, patient_dir, patient, mask_dir, mask,dicom_dir):
    """Create and save Sagittal plane binary mask image from .csv Osirix annotations.

    Args:
        img_types (List): List of considered
         Sagittal image types.
        patient_dir (str): Patient annotation directory path.
        patient (str): Patient id.
        mask_dir (str): Output mask directory path.
    """
    final_masks = {}
    supraspinatus_masks = {}
    teres_minor_masks = {}
    subscapular_masks = {}
    infraspinatus_masks = {}
    for img_type in img_types:
        # Get annotations
        new_shape = (512,512)
        annot_path = os.path.join(patient_dir, img_type, img_type + ".csv")
        if patient.startswith('A01') and patient !="A0103" or patient =="F0028" or patient == "A0098" or (patient == "F0015" and img_type=="T1_sag"):
            annotations = NewOsirixAnnotationList()
            patient_dir = os.path.join(dicom_dir, patient)
            images = list(set(os.listdir(patient_dir)) & set(img_types))
            for image in list(set(images)):
                dicom_path = os.path.join(patient_dir,image)
                shape = dimension_dicom(dicom_path,0)
        else:
            annotations = OsirixAnnotationList()
            shape = (512,512)

        annotations.load_from_csv(annot_path)

        muscles = []
        center_X_pts = []
        center_Y_pts = []
        for annotation in annotations.annotations:
            X, Y = np.array(annotation.px_X), np.array(annotation.px_Y)
            
            if X.shape != Y.shape or X.size == 0 or Y.size == 0:
                continue

            annotation_points = np.dstack((X, Y)).astype(int)
            # Create mask
            mask = create_mask(annotation_points, shape, new_shape)
            # if len(X) == 1: # TODO uncomment if you want to keep small tear annotations
            #    masks["tear"].append(mask)

            if patient.startswith("A01") and patient !="A0103" or patient =="F0028" or patient == "A0098" or (patient == "F0015" and img_type=="T1_sag"):
                if len(X) == 2:
                    final_masks["tangent_sign"] = mask
                if len(X) > 2:
                    muscles.append(mask)
                    M = cv2.moments(mask)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    center_X_pts.append(cX)
                    center_Y_pts.append(cY)

            else:
                if len(X) == 2:
                    final_masks["tangent_sign"] = mask
                if len(X) > 2:
                    if img_type == "T1_sag":
                        supraspinatus_masks[img_type] = mask
                    
                    else:
                        muscles.append(mask)
                        M = cv2.moments(mask)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        center_X_pts.append(cX)
                        center_Y_pts.append(cY)
       
                  
        if len(muscles) > 0:
                if len(muscles) >= 4:
                    supraspinatus_idx = center_Y_pts.index(min(center_Y_pts))
                    teres_minor_idx = center_Y_pts.index(max(center_Y_pts))
                    subscapular_idx = center_X_pts.index(min(center_X_pts))
                    supraspinatus_masks[img_type] = muscles[
                        supraspinatus_idx
                    ]  # Highest center
                    teres_minor_masks[img_type] = muscles[teres_minor_idx]  # Lowest center
                    subscapular_masks[img_type] = muscles[
                        subscapular_idx
                    ]  # Most left center
                    last_idx = list(
                        set([0, 1, 2, 3])
                    -   set([supraspinatus_idx, teres_minor_idx, subscapular_idx])
                    )  # Last center left
                    infraspinatus_masks[img_type] = muscles[last_idx[0]]

            # When teres_minor is missing
                elif len(muscles) == 3:
                    supraspinatus_masks[img_type] = muscles[
                        center_Y_pts.index(min(center_Y_pts))
                    ]  # Highest center
                    subscapular_masks[img_type] = muscles[
                        center_X_pts.index(min(center_X_pts))
                    ]  # Most left center
                    infraspinatus_masks[img_type] = muscles[
                        center_X_pts.index(max(center_X_pts))
                    ]  # Most right center

            # Undefined annotation configuration to verify manually
                else:
                    print(
                        "Verify muscle annotations of patient {}, image type: {}".format(
                            patient, img_type
                        )
                    )


    # List all muscle masks
    dict_masks = {}
    if len(supraspinatus_masks) > 0:
        dict_masks["supraspinatus"] = supraspinatus_masks
    if len(teres_minor_masks) > 0:
        dict_masks["teres_minor"] = teres_minor_masks
    if len(subscapular_masks) > 0:
        dict_masks["subscapular"] = subscapular_masks
    if len(infraspinatus_masks) > 0:
        dict_masks["infraspinatus"] = infraspinatus_masks

    # Select best segmentations for muscles,
    # according to Md. Kolo expertise (T1 fat > T1 in phase > T1)
    if len(dict_masks) > 0:
        for muscle, muscle_mask in dict_masks.items():
            keys = muscle_mask.keys()
            if "T1_fat_sag" in keys:
                final_masks[muscle] = muscle_mask["T1_fat_sag"]
            elif "T1_in_phase_sag" in keys:
                final_masks[muscle] = muscle_mask["T1_in_phase_sag"]
            if "T1_sag" in keys:
                final_masks[muscle] = muscle_mask["T1_sag"]

    # Save masks
    for key, value in final_masks.items():
        
        tmp_mask_dir = os.path.join(mask_dir, key)
        os.makedirs(tmp_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(tmp_mask_dir, patient + ".jpg"), value)
        # value_flip = cv2.flip(value, 1) # Flip mask vertically
        # cv2.imwrite(os.path.join(tmp_mask_dir, patient + "_flip.jpg"), value_flip)


def save_masks_cor(img_types, patient_dir, patient, mask_dir):
    """Create and save Coronal plane binary mask image from .csv Osirix annotations.

    Args:
        img_types (List): List of considered Coronal image types.
        patient_dir (str): Patient annotation directory path.
        patient (str): Patient id.
        mask_dir (str): Output mask directory path.
    """
    masks = {}

    for img_type in img_types:
        # Get annotations
        annot_path = os.path.join(patient_dir, img_type, img_type + ".csv")
        annotations = OsirixAnnotationList()
        annotations.load_from_csv(annot_path)

        for annotation in annotations.annotations:
            X, Y = np.array(annotation.px_X), np.array(annotation.px_Y)
            annotation_points = np.dstack((X, Y)).astype(int)

            # Create mask
            mask = create_mask(annotation_points, SHAPE)
            if is_circle(X, Y):
                masks["best_fit_circle"] = mask

            # TODO : Also select triangle mask?

        # Save masks
    for key, value in masks.items():
        tmp_mask_dir = os.path.join(mask_dir, key)
        os.makedirs(tmp_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(tmp_mask_dir, patient + ".jpg"), value)
