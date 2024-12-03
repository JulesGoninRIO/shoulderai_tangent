# Imports
import os

import cv2
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import random
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from utils.constants import *
from utils.utils_annot import create_new_annot_dir, save_annot_to_mask
from utils.utils_dicom import (
    create_new_dicom_dir,
    write_dicom_to_jpg_all_slices,
    write_dicom_to_jpg_ROI_slice,
)
from utils.utils_label import (
    save_labels_all_slices,
    save_labels_ROI_slice_cor,
    save_labels_ROI_slice_sag,
    save_split_labels,
    save_split_labels_crossval
)
from utils.utils_preproc import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   
        torch.backends.cudnn.benchmark = False



class ROIDataModule(pl.LightningDataModule):
    """Data module for ROI slice."""
    set_seed(42)
    def __init__(
        self,
        plane,
        img_type,
        mask="tangent_sign",
        batch_size=16,
        num_workers=4,
        val_perc=0.2,
        test_perc=0.1,
        k_folds = 6,
        root_dir="../data",
        original_dicom_dir=ORIGINAL_DATA_DIR + "/dicom/original",
        new_dicom_dir=ORIGINAL_DATA_DIR + "/dicom/new_dicom_2024",
        original_annot_dir=ORIGINAL_DATA_DIR + "/annotations/original",
        new_annot_dir=ORIGINAL_DATA_DIR + "/annotations/new_annot_2024",
        original_labels_path=ORIGINAL_DATA_DIR + "/labels.csv",
        fold_index = 0, 
        train_size = 1, 
        hyper_param_tuning = False, 
        val_index = 0
    ):
        super().__init__()
        assert plane in PLANE_TYPES
        if plane == "sagittal":
            assert img_type in SAG_IMAGE_TYPES
        else:
            assert img_type in COR_IMAGE_TYPES
        self.plane = plane
        self.img_type = img_type
        self.mask = mask
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_perc = test_perc
        self.val_perc = val_perc
        self.k_folds = k_folds
        self.fold_index = fold_index
        self.train_indices = None
        self.val_indices = None
        self.train_dataset= None
        self.val_dataset = None

        self.original_dicom_dir = original_dicom_dir
        self.original_annot_dir = original_annot_dir
        self.original_labels_path = original_labels_path

        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self.data_dir = os.path.join(root_dir, plane)
        self.split_labels_path = os.path.join(root_dir, f"split_labels.csv")
        self.new_dicom_dir = new_dicom_dir
        self.new_annot_dir = new_annot_dir

        self.main_images_plane = (
            SAG_IMAGE_MAIN_TYPES if plane == "sagittal" else COR_IMAGE_MAIN_TYPES
        )
        self.add_images_plane = (
            SAG_IMAGE_ADDITIONAL_TYPES
            if plane == "sagittal"
            else COR_IMAGE_ADDITIONAL_TYPES
        )
        self.mask_images_plane = (
            SAG_IMAGE_MASK_TYPES if plane == "sagittal" else COR_IMAGE_MAIN_TYPES
        )

        self.train_transform = get_transform_seg(phase="train", img_type=self.img_type, mask = self.mask)
        self.test_transform = get_transform_seg(phase="test", img_type=self.img_type, mask = self.mask)
        self.base_seed = 42
        self.call_prepare = False
        self.train_size = train_size
        self.hyper_param_tuning = hyper_param_tuning
        self.val_index = val_index
        self.set_fold(fold_index)
        self.prepare_data()
        

    def prepare_data(self):
        # From original DICOM to organized dicom folders
        # Checking if prepare_data was called before (pl.Trainer calls it by itself)
        if not self.call_prepare:
            create_new_dicom_dir(
                original_dir=self.original_dicom_dir, new_dir=self.new_dicom_dir
            )

            # From original annotations to organized annotation folders
            create_new_annot_dir(
                original_dir=self.original_annot_dir, new_dir=self.new_annot_dir
            )

            # Define train test labels per patient for sagittal images (if it does not exist yet)
            # 10 folds to test over the whole set 
            save_split_labels_crossval(
                dicom_dir=self.new_dicom_dir,
                root_dir=self.root_dir,
                n_splits=self.k_folds
            )

            # From DICOM images to jpg
            write_dicom_to_jpg_ROI_slice(
                    dicom_dir=self.new_dicom_dir,
                    annot_dir=self.new_annot_dir,
                    data_dir=self.data_dir,
                    img_types=SAG_IMAGE_MAIN_TYPES,
                    additional_img_types=self.add_images_plane,
                    split_labels_path=self.split_labels_path,
                    mask = self.mask
                )
            

            # From .csv annotations to mask images
            save_annot_to_mask(
                    annot_dir=self.new_annot_dir,
                    data_dir=self.data_dir,
                    plane=self.plane,
                    img_types=self.mask_images_plane,
                    split_labels_path=self.split_labels_path,
                    mask = self.mask,
                    dicom_dir = self.new_dicom_dir
                )
            images_dir = os.path.join(self.data_dir, "images", self.img_type)
            masks_dir = os.path.join(self.data_dir, "masks", self.img_type, self.mask)
            add_masks_dir = None
            if self.mask == "tangent_sign":
                add_masks_dir = os.path.join(self.data_dir, "masks",self.img_type, "supraspinatus",)

            augment_images_and_masks_in_place(images_dir, masks_dir, add_masks_dir)
            self.call_prepare = True

    def setup(self, stage=None):

        if stage == "fit":
            train_data = ROIDataset(
                data_dir=self.data_dir,
                img_type=self.img_type,
                mask=self.mask,
                phase="train",
                split_labels_path=self.split_labels_path,
                transform=self.train_transform,
                thicken_masks=True,
            )
            if self.hyper_param_tuning:
                # For hyperparameter tuning we use 5 fold cross-validation over training set
                kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.base_seed)
                splits = list(kf.split(range(len(train_data))))
                self.train_indices, self.val_indices = splits[self.val_index]
            else:
                # Define validation set for early stopping 
                self.train_indices, self.val_indices = train_test_split(range(len(train_data)), train_size = 0.8*self.train_size,test_size= 0.2 * self.train_size, random_state=self.base_seed, shuffle=True)
                
            self.train_dataset = Subset(train_data, self.train_indices)
            self.val_dataset = Subset(train_data, self.val_indices)

        if stage == "test":
            self.test_dataset = ROIDataset(
                data_dir=self.data_dir,
                img_type=self.img_type,
                mask=self.mask,
                phase="test",
                split_labels_path=self.split_labels_path,
                transform=self.test_transform,
                thicken_masks=True,
            )


    def set_fold(self, fold_index):
        self.fold_index = fold_index
        self.split_labels_path = os.path.join(self.root_dir, f"split_labels_fold_{self.fold_index}.csv")

        

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,

        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return dataloader



class ROIDataset(Dataset):
    """Dataset for ROI segmentation based on split labels CSV."""

    def __init__(self, data_dir,                # A.RandomGamma(gamma_limit=(95,105),p=0.2),
 img_type, mask, phase, split_labels_path, transform=None, thicken_masks=True):
        """
        Args:
            data_dir (str): Path to the directory containing the images.
            img_type (str): Image type (e.g., 'T1_sag').
            mask (str): Mask type.
            phase (str): 'train' or 'test', to specify which part of the dataset to load.
            split_labels_path (str): Path to the CSV file that contains patient splits.
            transform (callable, optional): Optional transform to be applied on a sample.
            thicken_masks (bool, optional): Whether to thicken mask lines. Default is True.
        """
        self.data_dir = data_dir
        self.img_type = img_type
        self.mask = mask
        self.transform = transform
        self.thicken_masks = thicken_masks

        # Read the split_labels.csv to filter patients by phase (train/test)
        if split_labels_path != None:
            df_split_labels = pd.read_csv(split_labels_path)
            self.patients = df_split_labels[df_split_labels["phase"] == phase]["patient"].values
            self.patients = [f"{name}.jpg" for name in self.patients] +  [f"{name}_aug.jpg" for name in self.patients]
        # The line below is for adding patients with T1_sag_in_phase 
        # self.patients = [f"{name}.jpg" for name in self.patients] +  [f"{name}_2.jpg" for name in self.patients]
        
        self.img_dir = os.path.join(data_dir, "images", img_type)
        self.mask_dir = os.path.join(data_dir, "masks",img_type, mask)
        if split_labels_path != None:
            dataset = list(set(self.patients) & set(os.listdir(self.img_dir)) & set(os.listdir(self.mask_dir)))
        else:
            dataset = list(set(os.listdir(self.img_dir)) & set(os.listdir(self.mask_dir)))
        self.dataset = dataset
        self.patient_map = {p: i for i, p in enumerate(self.dataset)}  # Map patient ID to index

        print(f"Number of patients in {phase} set: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient_id = self.dataset[idx]
        img_path = os.path.join(self.img_dir, patient_id)
        mask_path = os.path.join(self.mask_dir, patient_id)

        # Load image and mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_path = os.path.join(self.data_dir, "images", "T1_sag",patient_id)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if mask is None:
            mask_path = os.path.join(self.data_dir, "masks", "T1_sag", self.mask, patient_id)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        
        # mask = interpolate_line(mask)
        if self.thicken_masks:
            mask = thicken_lines(mask)

        # Normalize mask values from {0, 255} to {0, 1}
        mask = (mask / 255).astype(int)

        sample = {"image": img, "mask": mask, "image_orig": img}

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            sample["image"], sample["mask"] = transformed["image"], transformed["mask"]
        return sample 

    def get_patient(self, idx):
        if idx >= len(self.dataset):
            print(f"Index {idx} out of range (dataset length: {len(self.dataset)})")
            return None
        return self.dataset[idx]
    
    def get_index_by_patient_id(self, patient_id):
        """Return the index for a given patient ID."""
        try:
            idx = self.dataset.index(f"{patient_id}.jpg")
            return idx
        except ValueError:
            print(f"Patient {patient_id} not found in the dataset.")
            return None
