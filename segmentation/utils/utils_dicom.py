import os
from collections import defaultdict
from distutils.dir_util import copy_tree

import pandas as pd
import pydicom
import cv2
from src.annotations_model import*
from src.dicom_model import DirectDicomImport
from utils.constants import PLANE_TYPES

# Some patients have double sequences in one study.
# Those acquisition times are the ones to be selected in those cases
acquisition_time_patients_with_double_sequences = {
    "F0007": ("150607", ["T1_in_phase_sag", "T1_fat_sag", "T1_water_sag"]),
    "F0012": ("150354", ["T2_fs_cor"]),
    "F0015": ("153550", ["T1_fs_cor", "T2_fs_cor"]),
    "F0027": ("145831", ["T2_fs_sag"]),
    "F0039": ("114251", ["T1_in_phase_sag", "T1_fat_sag", "T1_water_sag"]),
    "F0044": ("161022", ["T1_in_phase_sag", "T1_fat_sag", "T1_water_sag"]),
    "F0046": ("111958", ["T2_fs_cor"]),
    "F0050": ("094431", ["T1_in_phase_sag", "T1_fat_sag", "T1_water_sag"]),
    "F0063": ("114528", ["T2_fs_cor"]),
    "F0070": ("170858", ["T1_in_phase_sag", "T1_fat_sag", "T1_water_sag"]),
    "F0077": ("122921", ["T1_fs_cor"]),
    "F0084": ("113316.015362", ["T2_fs_sag"]),
}

# Remove these dicoms because annotations were done on the wrong sequence
dicoms_to_remove = {
    "F0007": ["T1_in_phase_sag"],
    "F0013": ["T1_in_phase_sag", "T1_sag"],
    "A0001": ["T1_cor"],
}


def create_new_dicom_dir(original_dir, new_dir):
    """Create from original dicom directory a new directory with more descriptive file names.

    Args:
        original_dir (str): Original dicom directory path.
        new_dir (str): New dicom directory path.
    """
    exists = check_and_create_dir(new_dir)
    if exists:
        return

    for patient in os.listdir(original_dir):
        if patient.startswith('F') or patient.startswith('A'): 
            patient_dir = os.path.join(original_dir, patient)
            list_studies = os.listdir(patient_dir)

            for study in list_studies:
                tmp_dir = os.path.join(patient_dir, study)

                for sequence_name in os.listdir(tmp_dir):
                    slices_path = os.path.join(tmp_dir, sequence_name)

                # Get Sequence's description for first slice
                    slice_path = os.path.join(slices_path, os.listdir(slices_path)[0])
                # Check if it is a file
                    if os.path.isfile(slice_path):
                        ds = pydicom.filereader.dcmread(slice_path)
                        image_name = ds[0x0008103E].value.lower()
                        accession_number = ds[0x00080050].value  # AISHOULDER0085
                        if (0x0008, 0x0032) in ds:  # Verify if tag exists
                            acquisition_time = str(ds[0x00080032].value)
                        else:
                            acquisition_time = None

                # Get patient code (if there's two studies for same patient: F0085, F0085-2)
                    if len(list_studies) > 1:
                        patient = patient[0] + str(accession_number).replace(
                            "AISHOULDER", ""
                        )
                    else:
                        patient = patient

                    images_dir = os.path.join(new_dir, patient)
                    os.makedirs(images_dir, exist_ok=True)
                    new_name, new_images_dir = get_new_image_name_and_dir(
                        image_name, images_dir
                    )

                # If patient has double sequence, only select the right one
                    (
                        correct_acquisition_time,
                        list_sequences,
                    ) = acquisition_time_patients_with_double_sequences.get(
                        patient, ("", [])
                    )
                    if (
                        new_name in list_sequences
                        and acquisition_time != correct_acquisition_time
                    ):
                        continue

                # Verify if patient has dicoms to remove
                    if new_name in dicoms_to_remove.get(patient, []):
                        continue

                    new_image_path = os.path.join(new_images_dir, new_name)
                    copy_tree(slices_path, new_image_path)


def write_dicom_to_jpg_all_slices(
    dicom_dir, data_dir, phase, img_types, split_labels_path
):
    """From the dicom images, create the JPG images of each slices of each MRI for classification.

    Args:
        dicom_dir (str): Dicom directory path.
        data_dir (str): Newly generated data directory path, where JPG output will be written.
        phase (str):  Model phase of images considered. Either "train" or "test".
        img_types (str): Image types considered per plane:
            Sagittal: ['T1_sag', 'T1_in_phase_sag'], Coronal: ['T2_fs_cor'].
        split_labels_path (str): Train / test labels path.
    """
    jpg_dir = os.path.join(data_dir, phase)
    exists = check_and_create_dir(jpg_dir)
    if exists:
        print("All slices in JPG already exist for the {} set".format(phase))
        return

    # Select patients from the right model phase
    df_split_labels = pd.read_csv(split_labels_path)
    phase_patients = df_split_labels.loc[
        df_split_labels["phase"] == phase, "patient"
    ].values

    # Select patients from the right model phase and that have dicoms
    patients = list(set(phase_patients) & set(os.listdir(dicom_dir)))

    # Loop through patients to retrieve JPG images
    for patient in patients:
        patient_dir = os.path.join(dicom_dir, patient)
        images = list(set(os.listdir(patient_dir)) & set(img_types))

        for image in images:
            tmp_image_dir = os.path.join(patient_dir, image)
            images = DirectDicomImport(tmp_image_dir)
            images.read_files()
            np_images = images.images[0].as_numpy_array()

            tmp_jpg_dir = os.path.join(jpg_dir, image)
            os.makedirs(tmp_jpg_dir, exist_ok=True)

            for i in range(len(np_images)):
                im = np_images[i]
                im = resize_image(im, (512, 512))
                cv2.imwrite(
                    os.path.join(tmp_jpg_dir, "{}_{}.jpg".format(patient, i)), im
                )


def write_dicom_to_jpg_ROI_slice(
    dicom_dir,
    annot_dir,
    data_dir,
    img_types,
    additional_img_types,
    split_labels_path,
    mask,
):
    """Save DICOM images into a unified 'images' folder and their corresponding masks into a 'masks' folder."""
    
    images_dir = os.path.join(data_dir, "images")  # Unified images folder
    masks_dir = os.path.join(data_dir, "masks")    # Unified masks folder
    # Create directories if they don't exist
    if os.path.exists(images_dir):
        return
    os.makedirs(images_dir, exist_ok=True)
    
    # Select patients from the right model phase
    df_split_labels = pd.read_csv(split_labels_path)
    patients = list(set(df_split_labels["patient"].values) & set(os.listdir(dicom_dir)))
    
    # Loop through patients to retrieve images and corresponding masks
    for patient in patients:        
        patient_dir = os.path.join(dicom_dir, patient)
        images = list(set(os.listdir(patient_dir)) & set(img_types))
        patient_annot_dir = os.path.join(annot_dir, patient)        # Skip patients without annotations
        if not os.path.exists(patient_annot_dir):
            continue
        
        # Process the images and their corresponding masks
        for image in images:
            tmp_dicom_dir = os.path.join(patient_dir, image)
            tmp_annot_dir = os.path.join(patient_annot_dir, image)
            tmp_annot_path = os.path.join(tmp_annot_dir, "{}.csv".format(image))
            plane = PLANE_TYPES[0] if "sag" in image else PLANE_TYPES[1]
            ROI_coord = get_ROI_coord(tmp_annot_path, plane, patient)
            image_type = image
            patient_id = patient
            if ROI_coord is not None:
                # Save the images and masks in the unified folders
                save_ROI_jpg(tmp_dicom_dir, images_dir, ROI_coord, image_type, patient_id, mask)
            else:
                print(f"ROI coordinates not found for patient {patient}, image {image}")




def get_ROI_coord(annot_path, plane,patient):
    """Getter function for the ROI slice coordinate of an MRI.

    Args:
        annot_path (str): Considered .csv annotation path.
        plane (str): Plane of images and masks considered.
            Either "sagittal" or "coronal".

    Returns:
        Int: ROI slice coordinate.
    """
    max_layer = None
    image_type = os.path.splitext(os.path.basename(annot_path))[0]
    if os.path.isfile(annot_path):
        if patient.startswith("A01") and patient !="A0103" or patient =="F0028" or patient == "A0098" or (patient == "F0015" and image_type=="T1_sag"):
            annotations = NewOsirixAnnotationList()
        else:
            annotations = OsirixAnnotationList()
        annotations.load_from_csv(annot_path)

        if plane == PLANE_TYPES[0]:
            layers = defaultdict(int)
            for annot in annotations.annotations:
                layers[annot.image_no.value] += 1
            if layers:
                max_layer = max(layers, key=layers.get)
        else:
            max_layer = ...  # TODO adapt for coronal images
    return max_layer


def save_ROI_jpg(dicom_dir, jpg_dir, ROI_coord, img_type, patient,mask):
    """Read and save .jpg ROI slice image in jpg_dir.

    Args:
        dicom_dir (str): Newly generated dicom directory path.
        jpg_dir (str): Output directory path for JPG image.
        ROI_coord (Int): Coordinate of ROI slice.
        img_type (str): Image type considered.
        patient (str): Patient id.
    """
    if patient.startswith("A") and mask in ["subscapular", "infraspinatus", "teres_minor"]:
        img_type = "T1_in_phase_sag"
    images = DirectDicomImport(dicom_dir)
    images.read_files()
    if ROI_coord >= len(images.images[0].as_numpy_array()):
        ROI_coord = len(images.images[0].as_numpy_array()) - 1
        print("Some files of the study are missing for", patient)
    y_image = images.images[0].as_numpy_array()[ROI_coord]
    new_shape = (512,512)
    y_image = cv2.resize(y_image, new_shape,interpolation = cv2.INTER_LINEAR)
    tmp_jpg_dir = os.path.join(jpg_dir, img_type)
    os.makedirs(tmp_jpg_dir, exist_ok=True)
    cv2.imwrite(os.path.join(tmp_jpg_dir, "{}.jpg".format(patient)), y_image)
    # y_image_flip = cv2.flip(y_image, 1) # flip the image vertically
    # cv2.imwrite(os.path.join(tmp_jpg_dir, "{}_flip.jpg".format(patient)), y_image_flip)

    

def dimension_dicom(dicom_dir, ROI_coord):
   
    images = DirectDicomImport(dicom_dir)
    images.read_files()
    y_image = images.images[0].as_numpy_array()[ROI_coord]
    shape = y_image.shape
    return shape
       

def check_and_create_dir(dir_path):
    """Helper function to check whether a directory exists and is non-empty.
    If it doesn't exist create the directory.

    Args:
        dir_path (str): Directory path to check.

    Returns:
        Boolean: True if it exists and is non-empty, False otherwise.
    """
    exists = False
    if os.path.exists(dir_path):
        if len(os.listdir(dir_path)) > 0:
            exists = True
    if not exists:
        os.makedirs(dir_path, exist_ok=True)
        exists = False
    return exists


def get_new_image_name_and_dir(original_name, output_dir):
    """Getter function for new descriptive image name.

    Args:
        original_name (str): Original image name to replace.
        output_dir (str): Output directory path.

    Returns:
        Tuple: New image name and updated output directory.
    """
    # T1:
    if "t1" in original_name:
        if "sag" in original_name:
            if "phase" in original_name:
                new_name = "T1_in_phase_sag"

            elif "fat" in original_name:
                new_name = "T1_fat_sag"

            elif "water" in original_name:
                new_name = "T1_water_sag"

            elif "fs" in original_name:
                new_name = "T1_fs_sag"

            else:
                new_name = "T1_sag"

        elif "cor" in original_name:
            if len(original_name.split()) < 4:  # T1 cor or T1 cor propeller, ...
                new_name = "T1_cor"

            elif "fs" in original_name:
                new_name = "T1_fs_cor"

            else:
                output_dir = output_dir + "/other"
                new_name = original_name

        else:
            output_dir = output_dir + "/other"
            new_name = original_name

    # T2:
    elif "t2" in original_name:
        if "sag" in original_name and "fs" in original_name:
            new_name = "T2_fs_sag"

        elif "cor" in original_name and "fs" in original_name:
            new_name = "T2_fs_cor"

        else:
            output_dir = output_dir + "/other"
            new_name = original_name

    # Osirix annotations. Only for dicoms, not for annotations
    elif "osirix" in original_name:
        new_name = "annotations/" + original_name

    else:
        output_dir = (
            output_dir + "/other"
        )  # Update directory path as we add a new directory inside
        new_name = original_name

    return new_name, output_dir


def resize_image(image, size):
    """Redimensionne l'image à la taille spécifiée.

    Args:
        image (ndarray): Image à redimensionner.
        size (tuple): Nouvelle taille (largeur, hauteur).

    Returns:
        ndarray: Image redimensionnée.

    """
    if image is None:
        raise ValueError("The provided image is empty or None.")
    if image.size == 0:
        raise ValueError("The provided image has no data (size 0).")
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
