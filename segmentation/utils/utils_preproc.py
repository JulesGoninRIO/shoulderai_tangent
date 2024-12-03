import os

import albumentations as A
import numpy as np
import SimpleITK as sitk
import torch
from albumentations.pytorch.transforms import ToTensorV2
from intensity_normalization.normalize.nyul import NyulNormalize
from pystackreg import StackReg
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import random
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from albumentations.core.transforms_interface import DualTransform
import re 
from torchvision import transforms
from utils.utils_loss import get_line_end_points, prolong_line, write_infinite_line
import src.dataset as data
from utils.constants import (MEAN_STD_ALL_COR, MEAN_STD_ALL_SAG,
                             MEAN_STD_ROI_COR, MEAN_STD_ROI_SAG, PLANE_TYPES,
                             SHAPE)
from scipy.spatial.distance import cdist
from PIL import Image




class N4BiasFieldCorrection(A.ImageOnlyTransform):
    """Albumentation type transform for images only. Apply N4 bias field correction to image.
    See example:
    https://simpleitk.readthedocs.io/en/v2.1.0/link_N4BiasFieldCorrection_docs.html
    """

    def __init__(self, num_hist_bins=200):
        super().__init__()
        self.num_hist_bins = num_hist_bins

    def apply(self, img, **params):
        img = sitk.GetImageFromArray(img.astype(np.float32))
        mask_img = sitk.OtsuThreshold(img, 1, 0, self.num_hist_bins)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        img = corrector.Execute(img, mask_img)
        img_array = sitk.GetArrayFromImage(img)
        img_array = np.clip(img_array,0,255).astype(np.uint8)
        return img_array


class ContrastStretching(A.ImageOnlyTransform):
    """Albumentation type transform for images only. Apply contrast stretching
    to image.
    """

    def __init__(self, percentiles=[20, 100]):
        super().__init__()
        self.percentiles = percentiles

    def apply(self, img, **params):
        assert len(self.percentiles) == 2

        minmax_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
        img_clip = np.clip(
            img,
            np.percentile(img, self.percentiles[0]),
            np.percentile(img, self.percentiles[1]),
        )
        # Loop over the image and apply Min-Max formulae
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                minmax_img[i, j] = (img[i, j] - np.min(img_clip)) / (
                    np.max(img_clip) - np.min(img_clip)
                )
        print(f"contrast streching image type after transf : {type(minmax_img)}, dtyppe : {minmax_img.dtype}, shape: {minmax_img.shape}")
        return minmax_img


class NyulNormalization(A.ImageOnlyTransform):
    """Albumentation type transform for images only. Apply Nyul normalization to images,
    from precomputed standard histogram.
    See example:
    https://notebook.community/sergivalverde/MRI_intensity_normalization/Intensity%20normalization%20test
    """

    def __init__(self, img_type, hist_dir="./results/nyul_hist"):
        super().__init__()
        if not os.path.exists(hist_dir):
            if "sag" in img_type:
                save_histogram_nyul(PLANE_TYPES[0], hist_dir=hist_dir)
            else:
                save_histogram_nyul(PLANE_TYPES[1], hist_dir=hist_dir)
        self.hist_folder = hist_dir
        self.img_type = img_type

    def apply(self, img, **params):
        nyul_normalizer = NyulNormalize()

        # load the standard histogram
        nyul_normalizer.load_standard_histogram(
            os.path.join(
                self.hist_folder, "standard_histogram_{}.npy".format(self.img_type)
            )
        )
        img = nyul_normalizer(img)
        return img


class StackRegistration(DualTransform):
    """Albumentation type transform for images and masks.
    Registers stack according to precomputed registration matrix.
    See example: https://pystackreg.readthedocs.io/en/latest/readme.html
    """

    def __init__(self,img_type,mask, tmats_dir="./results/stackreg_tmats"):
        super().__init__()
        if mask is None:
            raise ValueError("mask error")
        self.mask = mask
        self.index = 0
        if not os.path.exists(tmats_dir):
            if "sag" in img_type:
                save_tmats_stackReg(PLANE_TYPES[0], mask, tmats_dir=tmats_dir)
            else:
                save_tmats_stackReg(PLANE_TYPES[1], mask, tmats_dir=tmats_dir)
        self.tmats = np.load(os.path.join(tmats_dir, "tmats_{}.npy".format(img_type)))

    def apply(self, img, mask = None, transf=StackReg.RIGID_BODY, **params):
        if mask is None:
            mask = self.mask
        sr = StackReg(transf)
        if not isinstance(img,np.ndarray):
            img = np.array(img)

        print(f"image type before transf : {type(img)}, dtyppe : {img.dtype}, shape: {img.shape}")
        img = img.astype(np.float32)    
        tmats = self.tmats[self.index]
        transformed_img = sr.transform(img, tmats)
        self.index +=1
        if not isinstance(transformed_img, np.ndarray):
            transformed_img = np.ndarray(transformed_img)
        transformed_img = transformed_img.astype(np.uint8)
        print(f"image type after transf : {type(transformed_img)}, dtyppe : {transformed_img.dtype}, shape: {transformed_img.shape}")
        return transformed_img
    
    def apply_to_mask(self, mask,transf = StackReg.RIGID_BODY, **params):
        """
        sr = StackReg(transf)
        if not isinstance(mask,np.ndarray):
            mask = np.array(mask)
        
        print(f"mask type before transf : {type(mask)}, dtyppe : {mask.dtype}, shape: {mask.shape}")
        mask = mask.astype(np.float32)
        
        tmats = self.tmats[self.index]
        transformed_mask = sr.transform(mask, tmats)
          
        if not isinstance(transformed_mask, np.ndarray):
            transformed_mask = np.ndarray(transformed_mask)
        self.index +=1
        transformed_mask = transformed_mask.astype(np.float32)
        print(f"mask type after transf : {type(transformed_mask)}, dtyppe : {transformed_mask.dtype}, shape: {transformed_mask.shape}")
        """
        return mask
        


def get_transform_classif(phase, img_type):
    """Getter function for image transform of classification model. MEAN and STD values for the considered
    image type are read from utils/constants.py and were computed with the function get_mean_std() below.

    Args:
        phase (str): Either "train" or "test".
        img_type (str): Considered image type.

    Returns:
        Compose: Composition of transforms.
    """
    if img_type.endswith("sag"):
        (MEAN, STD) = MEAN_STD_ALL_SAG[img_type]
    if img_type.endswith("cor"):
        (MEAN, STD) = MEAN_STD_ALL_COR[img_type]

    # TODO adapt transform to image type
    if phase == "train":
        transform = A.Compose(
            [
                A.Resize(SHAPE[0], SHAPE[1]),
                A.CLAHE(),
                # StackRegistration(img_type=img_type),
                # A.ShiftScaleRotate(),
                A.CenterCrop(height=256, width=256),
                # A.HorizontalFlip(p=0.5),
                A.Normalize(MEAN, STD),
                ToTensorV2(),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.Resize(SHAPE[0], SHAPE[1]),
                A.CLAHE(),
                # StackRegistration(img_type),
                A.Normalize(MEAN, STD),
                ToTensorV2(),
            ]
        )

    return transform

class Filter(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__()
    def apply(self, img, **params):
        kernel = np.ones((11,11), np.uint8)
        black_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        return black_img


def thicken_lines(mask, thickness = 11):
    """Apply dilation to thicken lines in the mask."""
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()  # En
    mask = mask.astype(np.uint8)
    kernel = np.ones((thickness, thickness), np.uint8)
    # Apply dilation
    thick_mask = cv2.dilate(mask, kernel, iterations=1)
    return thick_mask

def thin_lines(mask, thickness=11):
    """Apply erosion to thin lines in the mask."""
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    kernel = np.ones((thickness, thickness), np.uint8)
    # Apply erosion
    thin_mask = cv2.erode(mask, kernel, iterations=1)
    return thin_mask


def get_transform_seg(phase, img_type, mask):
    """Getter function for image transform of segmentation model. MEAN and STD values for the considered
    image type are read from utils/constants.py and were computed with the function get_mean_std() below.

    Args:
        phase (str): Either "train" or "test".
        img_type (str): Considered image type.

    Returns:
        Compose: Composition of transforms.
    """
    if img_type.endswith("sag"):
        (MEAN, STD) = MEAN_STD_ROI_SAG[img_type]
    if img_type.endswith("cor"):
        (MEAN, STD) = MEAN_STD_ROI_COR[img_type]

    # TODO adapt transform to image type
    if phase == "train":
        transform = A.Compose(
            [
                A.Resize(SHAPE[0], SHAPE[1]),
                N4BiasFieldCorrection(),
                #CropAroundMask(output_size = (256,256), p = 1),
                #A.CLAHE(clip_limit = 2.0, title_grid_size=(8,8), p=1),
                A.CLAHE(),
                A.Equalize(p=1),
                # StackRegistration(img_type, mask),
                # A.Rotate(limit=15, p=0.5),  
                # A.RandomGamma(gamma_limit=(95,105),p=0.2),
                A.Normalize(MEAN, STD),
                ToTensorV2(),
            ]
        )
    else:
        if mask == 'tangent_sign':
            transform = A.Compose(
                [
                    A.Resize(SHAPE[0], SHAPE[1]),
                    # N4BiasFieldCorrection(),
                    #CropAroundMask(output_size = (256,256), p = 1),
                    A.CLAHE(),
                    # StackRegistration(img_type),
                    #Skeletonize(always_apply=True),
                    # A.Equalize(p=1),
                    #A.RandomBrightnessContrast(brightness_limit=(0.2,0.5), contrast_limit = (0.2,0.5), p=1.0),
                    # A.VerticalFlip(p=1),
                    # A.ShiftScaleRotate(
                    # shift_limit_y=0.3,
                    # shift_limit_x=0,
                    # scale_limit=0.0,
                    # rotate_limit=0,
                    # p=1,
                    # ),
                    A.Normalize(MEAN, STD),
                    ToTensorV2(),
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Resize(SHAPE[0], SHAPE[1]),
                    #N4BiasFieldCorrection(),
                    A.CLAHE(),
                    #A.CLAHE(clip_limit = 2.0, title_grid_size=(8,8), p=1),
                    #StackRegistration(img_type, mask),
                    A.Normalize(MEAN, STD),
                    ToTensorV2(),
                ]
            )
    return transform


def save_tmats_stackReg(
    plane,
    mask,
    tmats_dir="./results/stackreg_tmats",
    data_dir="./data",
    transf_stackreg=StackReg.RIGID_BODY,
    reference='mean',
):
    """Save registration matrix for stack registration transform.

    Args:
        plane (str): Either "sagittal" or "coronal".
        tmats_dir (str, optional): Path to results directory.
            Defaults to "./results/stackreg_tmats".
        data_dir (str, optional): Path to data directory. Defaults to "./data".
        transf_stackreg (constant, optional): Type of registration to perform.
            Defaults to StackReg.RIGID_BODY.
        reference (str, optional): Image reference for registration of next
            image in stack. Defaults to "mean".
    """
    plane_dir = os.path.join(data_dir, plane)
    

    # TODO how to compute it?
    # Is it possible to use the mean of all train data?
    # Does it work with the shapes? Do per batch?
    mean_std_dict = get_mean_std("sagittal", "./data", mask)
    update_constants("./utils/constants.py", mean_std_dict, "sagittal")
    mean, std = mean_std_dict.get("sagittal", (0.0,1.0))
    img_list = []
    os.makedirs(tmats_dir, exist_ok=True)
    for slice_type in os.listdir(plane_dir):
        phase_dir = os.path.join(plane_dir, slice_type, "train")
        for img_type in tqdm(os.listdir(phase_dir)):
            transf = A.Compose([
                A.Resize(SHAPE[0], SHAPE[1]),
                A.CLAHE(),
                A.Normalize(mean = mean, std=std),
                ToTensorV2(),
            ])
            img_loader = get_simple_train_dataloader(
                img_type, slice_type, plane_dir,mask, batch_size=16, transf=transf
            )

            if img_loader:
                tmats = []
                sr = StackReg(transf_stackreg)
                for batch_idx, sample in enumerate(img_loader):
                    img = sample["image"].numpy()
                  
                    if isinstance(img, list):
                        img = np.array(img)
                    elif isinstance(img,torch.Tensor):
                        img = img.numpy()
                    
                    if img.ndim == 3:
                        pass
                    elif img.ndim == 4:
                        img = np.squeeze(img, axis = 1)

                    img_list.append(img)
                    print(f"batch {batch_idx} - Image shape befire regisration: {img.shape}")
                    try:
                        tmats_tmp = sr.register_stack(img, axis=0, reference=reference)
                        print(f"tmats_tmp shape after registration : {tmats_tmp.shape}")
                        print(f"image shape after registration : {img.shape}")
                        if tmats_tmp.shape[0] != img.shape[0]:
                            raise Exception("Number of transfromation matrics does not number of images")
                        for mat in tmats_tmp:
                            tmats.append(mat)
                        #tmats.append(tmats_tmp.mean(axis = 0))
                        print(f"appended tmats shape : {np.array(tmats).shape}")
                    except Exception as e:
                        print(f"Error during registration: {e}")
                        continue
                print(f"total number of images processed : {batch_idx + 1}")
                print(f"total nombre matrcies : {len(tmats)}")
                if tmats :
                    #tmats = np.mean(tmats,axis = 0)
                    tmats = np.array(tmats)
                    print(f"final tmats shape fro {img_type}: {tmats.shape}")
                    np.save(os.path.join(tmats_dir, f"tmats_{img_type}.npy"), tmats)
                else:
                    print("no valid image generated for {img_type}")
        img = np.concatenate(img_list, axis = 0)
        return img


                  
def save_histogram_nyul(plane, hist_dir="./results/nyul_hist", data_dir="./data"):
    """Save Nyul histogram for Nyul histogram equalization image transform.

    Args:
        plane (str): Either "sagittal" or "coronal".
        hist_dir (str, optional): Path to histogram directory. Defaults to "./results/nyul_hist".
        data_dir (str, optional): Path to data directory. Defaults to "./data".
    """
    plane_dir = os.path.join(data_dir, plane)
    transf = A.Compose(
        [
            A.Resize(SHAPE[0], SHAPE[1]),
            ToTensorV2()
        ]
    )
    os.makedirs(hist_dir, exist_ok=True)
    for slice_type in os.listdir(plane_dir):
        phase_dir = os.path.join(plane_dir, slice_type, "train")
        for img_type in tqdm(os.listdir(phase_dir)):
            img_loader = get_simple_train_dataloader(
                img_type, slice_type, plane_dir, batch_size=1, transf=transf
            )

            if img_loader:
                nyul_normalizer = NyulNormalize()
                imgs = []
                for sample in img_loader:
                    img = sample["image"]
                    imgs.append(img)

                # normalize the images and save the standard histogram
                nyul_normalizer.fit(imgs)
                nyul_normalizer.save_standard_histogram(
                    os.path.join(hist_dir, "standard_histogram_{}.npy".format(img_type))
                )


def get_mean_std_test(plane, data_dir, mask):

    plane_dir = os.path.join(data_dir, plane)
    transf = A.Compose(
        [
            A.Resize(SHAPE[0], SHAPE[1]),
            ToTensorV2(),
        ]
    )
    mean_std_dict_test = {}
    for slice_type in os.listdir(plane_dir):
        phase_dir = os.path.join(plane_dir, slice_type, "test")
        for img_type in os.listdir(phase_dir):
            if img_type != "T1_sag":
                mask_dir = os.path.join(phase_dir, 'masks', 'other_T1_sag', mask)
            else:
                mask_dir = os.path.join(phase_dir, 'masks', img_type, mask)
            if not os.path.exists(mask_dir):
                continue
            img_loader = get_simple_test_dataloader(
                img_type, slice_type, plane_dir,mask, batch_size=500, transf=transf
            )

            if img_loader:
                channel_sum, square_channel_sum, num_batches = (
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                )

                for sample in img_loader:
                    img = sample["image"] / 255
                    channel_sum += torch.mean(img, dim=[0, 2, 3])
                    square_channel_sum += torch.mean(img**2, dim=[0, 2, 3])
                    num_batches += 1

                mean = channel_sum / num_batches
                # var = E[x^2] - (E[x])^2
                var = square_channel_sum / num_batches - mean**2
                std = torch.sqrt(var)
                mean_std_dict_test[img_type] = (mean.item(),std.item())
        return mean_std_dict_test

def get_mean_std(plane, data_dir, mask):
    """Getter function for mean and standard pixel values on all train data for
    each image type of the selected plane. Used to standardize images.

    Args:
        plane (str): Either "sagittal" or "coronal".
        data_dir (str, optional): Path to data directory. Defaults to "./data".
    """

    plane_dir = os.path.join(data_dir, plane)
    transf = A.Compose(
        [
            A.Resize(SHAPE[0], SHAPE[1]),
            ToTensorV2(),
        ]
    )
    mean_std_dict_train = {}
    for slice_type in os.listdir(plane_dir):
        phase_dir = os.path.join(plane_dir, slice_type, "train")
        for img_type in os.listdir(phase_dir):
            img_loader = get_simple_train_dataloader(
                img_type, slice_type, plane_dir,mask, batch_size=500, transf=transf
            )

            if img_loader:
                channel_sum, square_channel_sum, num_batches = (
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                )

                for sample in img_loader:
                    img = sample["image"] / 255
                    channel_sum += torch.mean(img, dim=[0, 2, 3])
                    square_channel_sum += torch.mean(img**2, dim=[0, 2, 3])
                    num_batches += 1

                mean = channel_sum / num_batches
                # var = E[x^2] - (E[x])^2
                var = square_channel_sum / num_batches - mean**2
                std = torch.sqrt(var)

                mean_std_dict_train[img_type] = (mean.item(),std.item())

    return mean_std_dict_train


def mean_std_total(plane, data_dir, mask):

    mean_std_dict_train = get_mean_std(plane, data_dir, mask)
    mean_std_dict_test = get_mean_std_test(plane, data_dir, mask)
    mean_std_dict = {}
    for img_type in mean_std_dict_train:
            if img_type in mean_std_dict_test:
                mean_avg = (mean_std_dict_train[img_type][0] + mean_std_dict_test[img_type][0]) / 2
                std_avg = (mean_std_dict_train[img_type][1] + mean_std_dict_test[img_type][1]) / 2
                mean_std_dict[img_type] = (mean_avg, std_avg)
            else:
                mean_std_dict[img_type] = mean_std_dict_train[img_type]
    for img_type in mean_std_dict_train:
        if img_type in mean_std_dict_train:
            mean_std_dict[img_type] = mean_std_dict_test[img_type]
    return mean_std_dict



def get_simple_test_dataloader(
    img_type, slice_type, plane_dir, mask, batch_size=1, transf=None
):
    """Getter function for dataloader with basic settings to test other functions.

    Args:
        img_type (str): Considered image type.
        slice_type (str): Either "all_slices" or "ROI_slice".
        plane_dir (str): Path to images of considered plane.
        batch_size (int, optional): Batch size for dataloader. Defaults to 1.
        transf (Compose, optional): Transform to apply on dataset. Defaults to None.

    Returns:
        Dataloader: Dataloader with specified settings.
    """
    data_dir = os.path.join(plane_dir, slice_type)
    img_loader = None
    if img_type != "masks":
        if not transf:
            transf = (
                get_transform_classif("test", img_type)
                if slice_type == "all_slices"
                else get_transform_seg(phase="test", img_type=img_type)
            )
        if slice_type == "all_slices":
            dataset = data.AllSlicesDataset(
                data_dir=data_dir, img_type=img_type, phase="test", transform=transf
            )
        else:
            dataset = data.ROIDataset(
                data_dir=data_dir,
                img_type=img_type,
                mask= mask,
                phase="test",
                transform=transf,
            )

        img_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    return img_loader



def get_simple_train_dataloader(
    img_type, slice_type, plane_dir, mask, batch_size=1, transf=None
):
    """Getter function for dataloader with basic settings to test other functions.

    Args:
        img_type (str): Considered image type.
        slice_type (str): Either "all_slices" or "ROI_slice".
        plane_dir (str): Path to images of considered plane.
        batch_size (int, optional): Batch size for dataloader. Defaults to 1.
        transf (Compose, optional): Transform to apply on dataset. Defaults to None.

    Returns:
        Dataloader: Dataloader with specified settings.
    """
    data_dir = os.path.join(plane_dir, slice_type)
    img_loader = None
    if img_type != "masks":
        if not transf:
            transf = (
                get_transform_classif("train", img_type)
                if slice_type == "all_slices"
                else get_transform_seg(phase="train", img_type=img_type)
            )
        if slice_type == "all_slices":
            dataset = data.AllSlicesDataset(
                data_dir=data_dir, img_type=img_type, phase="train", transform=transf
            )
        else:
            dataset = data.ROIDataset(
                data_dir=data_dir,
                img_type=img_type,
                mask= mask,
                phase="train",
                transform=transf,
            )

        img_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    return img_loader

def update_constants(file_path, new_values,plane):
    with open(file_path, "r") as file:
        content = file.read()

    for img_type, (mean,std) in new_values.items():
        pattern = rf'"{img_type}": \([0-9\.\, ]+\)'
        replacement = f'"{img_type}" : ({mean}, {std})'
        content = re.sub(pattern, replacement, content)

    with open(file_path, "w") as file:
        file.write(content)



def save_image(pred, path):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    # Normalisation des images pour les enregistrer correctement
    pred = (pred > 0.5).astype(np.uint8)*255

    cv2.imwrite(path, pred)


def save_overlay_3_mask(image, pred, mask, mask_supra, path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(mask_supra, torch.Tensor):
        mask_supra = mask_supra.cpu().numpy()
    mask = thicken_lines(mask, thickness = 2)
    pred = thicken_lines(pred, thickness = 2)
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]

    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1,2,0))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    pred = cv2.resize(pred , (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)
    mask = cv2.resize(mask , (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)
    mask_supra = cv2.resize(mask_supra, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)

    mask_supra = (mask_supra >0.5).astype(np.uint8)*255
    mask = (mask >0.5).astype(np.uint8)*255
    pred = (pred >0.5).astype(np.uint8)*255
    
    overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)    
    overlay[..., 1] = pred  
    overlay[..., 2] = mask
    overlay[..., 0] = mask_supra 
    combined = cv2.addWeighted(image, 1, overlay, 1,0)
    cv2.imwrite(path, combined)


def save_overlay_image(image, pred, mask, path, supra_mask= None, color_pred = 1):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if supra_mask is not None:
        if isinstance(supra_mask, torch.Tensor):
            supra_mask = supra_mask.cpu().numpy()
            supra_mask = (supra_mask * 255).astype(np.uint8)

    mask = thicken_lines(mask, thickness = 3)
    pred = thicken_lines(pred, thickness = 3)

    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]

    pred = (pred * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    

    overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    if supra_mask is not None:
        overlay[..., 2] = supra_mask
    # overlay[..., 1] = pred  
    # overlay[..., 1] = np.maximum(overlay[..., 1],mask)
    overlay[..., color_pred] = pred
    overlay[..., 0] = mask 

    overlay = cv2.addWeighted(overlay, 1, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 1, 0)
    cv2.imwrite(path, overlay)


def skeletonize_mask(mask):
    if mask.dtype != bool:
        mask = mask > 0  # Convertir en booléen si ce n'est pas le cas
    # Appliquer la squelettisation
    skeleton = skeletonize(mask.astype(bool))
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)  # S'assurer que le résultat est en uint8
    # Convertir le squelette en binaire strict 0 ou 1
    return (skeleton_uint8 > 0).astype(np.uint8)


class CropAroundMask(DualTransform):
    def __init__(self, output_size=(256, 256), p=1.0):
        super(CropAroundMask, self).__init__(p=p)
        self.output_size = output_size

    def apply(self ,img, x1, y1, x2, y2, **params):
        return self._crop_and_pad(img, x1, y1, x2, y2)

    def apply_to_mask(self, mask, x1, y1, x2, y2, **params):
        return self._crop_and_pad(mask, x1, y1, x2, y2)

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        if mask is None or not np.any(mask):
            raise ValueError("Valid mask must be provided for CropAroundMask.")

        # Find the bounding box coordinates of the mask
        y_indices, x_indices = np.where(mask)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)

        # Calculate cropping boundaries
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        half_w, half_h = self.output_size[0] // 2, self.output_size[1] // 2

        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(mask.shape[1], center_x + half_w)
        y2 = min(mask.shape[0], center_y + half_h)

        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _crop_and_pad(self, img, x1, y1, x2, y2):
        cropped_img = img[y1:y2, x1:x2]

        # Add padding if the cropped image is smaller than the target size
        pad_top = max(0, self.output_size[1] - (y2 - y1))
        pad_bottom = max(0, self.output_size[1] - (y2 - y1) - pad_top)
        pad_left = max(0, self.output_size[0] - (x2 - x1))
        pad_right = max(0, self.output_size[0] - (x2 - x1) - pad_left)

        cropped_img = cv2.copyMakeBorder(
            cropped_img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Padding with black
        )

        return cropped_img
    
    def get_transform_init_args_names(self):
        return ("output_size",)

    def __call__(self, **kwargs):
        params = self.get_params()
        targets = {'mask': kwargs.get('mask')}
        dependent_params = self.get_params_dependent_on_targets(targets)
        x1, y1, x2, y2 = dependent_params["x1"], dependent_params["y1"], dependent_params["x2"], dependent_params["y2"]

        img = kwargs["image"]
        mask = kwargs["mask"]

        cropped_img = self.apply(img, x1, y1, x2, y2)
        cropped_mask = self.apply_to_mask(mask, x1, y1, x2, y2)

        return {"image": cropped_img, "mask": cropped_mask}
    


class DilateMask(DualTransform):
    def __init__(self, dilation_kernel_size=3, always_apply=False, p=1.0):
        super(DilateMask, self).__init__(always_apply, p)
        self.dilation_kernel_size = dilation_kernel_size

    def apply_to_mask(self, mask, **params):
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        return dilated_mask

    def get_transform_init_args_names(self):
        return ("dilation_kernel_size",)
    

def filter_segments(prediction, size_threshold=200, distance_threshold=300):
    """Filter our noise from prediction, removinf small segments and that are far away

    Args:
        prediction (tensor): output of the model
        size_threshold (int, optional): If segment is smaller, then it will be filtered out. Defaults to 200.
        distance_threshold (int, optional): If the distance of the segment to the biggest one is bigger, the it will be filtered out . Defaults to 300.

    Returns:
        tensor: filtered prediction
    """
    # Convert the prediction to a binary mask (0 and 255)
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255

    # Find all connected components
    num_labels, labels_im = cv2.connectedComponents(binary_mask)
    
    # Find the largest component
    if len(np.bincount(labels_im.flat)[1:]) == 0:
        return prediction
    largest_component_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    
    # Initialize an empty mask to keep the desired components
    final_mask = np.zeros_like(labels_im)
    
    # Get coordinates of the largest component
    largest_component_coords = np.column_stack(np.where(labels_im == largest_component_label))
    
    for label in range(1, num_labels):
        component_mask = (labels_im == label).astype(np.uint8)
        component_size = np.sum(component_mask)
        if component_size >= size_threshold:
            # Get coordinates of the current component
            component_coords = np.column_stack(np.where(component_mask == 1))
            
            # Calculate minimum distance to the largest component
            distances = cdist(component_coords, largest_component_coords, metric='euclidean')
            min_distance = np.min(distances)
            
            # Keep the component if it's within the distance threshold
            if min_distance <= distance_threshold or label == largest_component_label:
                final_mask[labels_im == label] = 255
    
    # Optionally, convert back to the original range (0 to 1)
    final_mask = final_mask / 255.0
    
    return final_mask


def fit_line(pred, extend= True ):
    """Function to fit the line for the prediction 

    Args:
        pred (tensor): prediction from the model
    
    Returns:
        tensor: mask with fitted line
    """

    # Ensure pred is a PyTorch tensor
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)

    pred_np = pred.cpu().numpy()
    y_indices, x_indices = np.where(pred_np == 1)# Threshold for positive pixels

    # Find connected components
    if len(y_indices) < 2:
        return pred

    # Fit a line to the points
    coeffs = np.polyfit(x_indices, y_indices, deg=1)
    linear_pred = np.zeros_like(pred_np)
    height , width = linear_pred.shape
    if extend:
        x_min = 0
        x_max = width - 1
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
    for x in range(x_min, x_max + 1):
        y = int(round(coeffs[0] * x + coeffs[1]))
        if 0 <= y < height:
            linear_pred[y, x] = 1.0
    return torch.tensor(linear_pred).to(pred.device)



def post_process_mask(pred):
    """Function to filter segments and extend mask"""
    
    # Ensure pred is a PyTorch tensor
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)

    pred_np = pred.cpu().numpy()
    y_indices, x_indices = np.where(pred > 0.5)  # Threshold for positive pixels

    # Find connected components
    if len(y_indices) < 2:
        return pred

    # Fit a line to the points
    coeffs = np.polyfit(x_indices, y_indices, deg=1)

    # Calculate original line endpoints
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = coeffs[0] * min_x + coeffs[1], coeffs[0] * max_x + coeffs[1]

    # Calculate the length of the original line
    original_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    # Extend the line by 1.5 times its original length
    extend_length = 1.5 * original_length

    # Calculate new extended endpoints
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    scale_factor = extend_length / original_length

    new_min_x = min_x - scale_factor * delta_x / 2
    new_max_x = max_x + scale_factor * delta_x / 2
    new_min_y = min_y - scale_factor * delta_y / 2
    new_max_y = max_y + scale_factor * delta_y / 2

    # Ensure the new coordinates are within the image bounds
    height, width = pred_np.shape
    new_min_x = max(0, min(width - 1, new_min_x))
    new_max_x = max(0, min(width - 1, new_max_x))
    new_min_y = max(0, min(height - 1, new_min_y))
    new_max_y = max(0, min(height - 1, new_max_y))

    # Generate the extended line
    linear_pred = np.zeros_like(pred_np)
    x_values = np.linspace(new_min_x, new_max_x, num=int(new_max_x - new_min_x) + 1)
    y_values = coeffs[0] * x_values + coeffs[1]

    # Fill in the line
    for x, y in zip(x_values, y_values):
        y = int(round(y))
        x = int(round(x))
        if 0 <= y < height and 0 <= x < width:
            linear_pred[y, x] = 1.0

    return torch.tensor(linear_pred).to(pred.device)



def post_process(pred, img_shape):
    
    pred = torch.tensor(pred, dtype = torch.float32)
    pt0, pt1 = pred , pred[-1]
    pto_prolo, pt1_prolo = prolong_line(pt0, pt1, img_shape)
    linear = np.zeros(img_shape, dtype = np.uint8)
    x0,y0 = pto_prolo.int()
    x1, y1 = pt1_prolo.int()
    cv2.line(linear, (x0,y0), (x1,y1),1,1)

    return torch.tensor(linear, dtype = torch.float32)
   

class Skeletonize(DualTransform):
    def __init__(self, always_apply=False):
        super(Skeletonize, self).__init__(always_apply)

    def apply_to_mask(self, mask, **params):
        # Assurez-vous que le masque est en format booléen pour la squelettisation
        if mask.dtype != bool:
            mask = mask > 0  # Convertir en booléen si ce n'est pas le cas

        # Appliquer la squelettisation
        skeleton = skeletonize(mask.astype(bool))
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)  # S'assurer que le résultat est en uint8

        # Convertir le squelette en binaire strict 0 ou 1
        return (skeleton_uint8 > 0).astype(np.uint8)

    def apply(self, img, **params):
        # Cette méthode n'est pas utilisée mais doit être présente pour compléter la classe.
        return img


def process_masks_for_loss(masks):
    """Process a batch of masks to convert them to 1-pixel wide lines."""
    processed_masks = []
    for mask in masks:
        processed_mask = skeletonize_mask(mask.cpu().numpy())
        processed_masks.append(torch.from_numpy(processed_mask).to(mask.device))
    return torch.stack(processed_masks)



def tensor_to_numpy(tensor):
    """
    Safely converts a PyTorch tensor to a numpy array.
    Checks if the input is a tensor before conversion.

    Args:
        tensor (torch.Tensor or numpy.ndarray): Input data that might be a tensor or already a numpy array.

    Returns:
        numpy.ndarray: A numpy array representation of the input.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()  # Ensure it's detached, moved to cpu, and then converted
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError("Input must be a PyTorch tensor or a numpy array")
    

def extended_mask(mask):
    pt0, pt1 = get_line_end_points(mask, 255)
    mask = (pt0, pt1, mask)
    return mask

def extend_line(image, start, end):
    """
    Extend a line from start to end points to the edges of the image.
    
    Args:
        image (torch.Tensor): The image tensor (2D).
        start (tuple): The starting point of the line (row, col).
        end (tuple): The ending point of the line (row, col).
    
    Returns:
        torch.Tensor: The image tensor with the extended line.
    """
    h, w = image.shape

    # Calculate the slope (rise/run)
    delta_y = end[0] - start[0]
    delta_x = end[1] - start[1]

    # Calculate the direction of the line
    if delta_x == 0:  # Vertical line
        slope = np.inf
    else:
        slope = delta_y / delta_x

    # Extend line to left edge
    if slope == np.inf:
        start_extended = (0, start[1])  # Extend vertically
        end_extended = (h-1, start[1])
    else:
        start_extended = (int(start[0] - start[1] * slope), 0)  # Extend to the left edge
        end_extended = (int(end[0] + (w - 1 - end[1]) * slope), w - 1)  # Extend to the right edge

    # Clip the coordinates to ensure they are within the image bounds
    start_extended = (max(0, min(h-1, start_extended[0])), max(0, min(w-1, start_extended[1])))
    end_extended = (max(0, min(h-1, end_extended[0])), max(0, min(w-1, end_extended[1])))

    # Draw the extended line
    rr, cc = draw_line(start_extended, end_extended)
    image[rr, cc] = 255

    return image
def draw_line(start, end):
    """
    Bresenham's line algorithm to get the points of a line.
    
    Args:
        start (tuple): The starting point of the line (row, col).
        end (tuple): The ending point of the line (row, col).
    
    Returns:
        list: Row indices of the line.
        list: Column indices of the line.
    """
    rr, cc = [], []
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        rr.append(x0)
        cc.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return rr, cc
    
def extended_mask1(pred, pt0, pt1, color = 255, thickness = 1):
    # Ensure pred is a PyTorch tensor
    height, width = pred.shape[:2]
    slope = (pt1[1]- pt0[1]/pt1[0]-pt0[0])
    intercept = pt1[1] - slope * pt1[0]

    
    y_intercept_right = int((slope) * (width -1) + intercept)
    y_intercept_left = int(intercept)
    
    points = []

    
    if  0 <= y_intercept_right < height:
        points.append((width - 1, y_intercept_right))
    if  0 <= y_intercept_left < height:
        points.append((0, y_intercept_left))

    if len(points) >= 2:
        cv2.line(pred, points[0], points[1], color, thickness)

    return pred


def load_images(folder, filenames=None, exclude_filenames=None):
    images = {}
    for filename in os.listdir(folder):
        file_id = os.path.splitext(filename)[0]  # Extract file ID without extension
        if (filenames is None or file_id in filenames) and (exclude_filenames is None or file_id not in exclude_filenames):
            filepath = os.path.join(folder, filename)
            img = cv2.imread(filepath)
            if img is not None:
                images[file_id] = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), filepath)
    return images

# Function to compute average histogram of a dataset
def compute_average_image(images):
    sum_image = None
    count = len(images)
    
    for img, _ in images.values():
        if sum_image is None:
            sum_image = np.zeros_like(img, dtype=np.float32)
        sum_image += img.astype(np.float32)
    
    avg_image = sum_image / count
    return avg_image.astype(np.uint8)

# Function to perform histogram matching
def histogram_matching(source_images, reference_histogram):
    matched_images = {}
    for file_id, (src_img, filepath) in source_images.items():
        matched = exposure.match_histograms(src_img, reference_histogram)
        matched_images[file_id] = (matched, filepath)
    return matched_images

# Paths to the folders containing the train and test images
def hist_match():
    train_folder = './data/sagittal/ROI_slice/train/T1_sag'
    test_folder = './data/sagittal/ROI_slice/test/T1_sag'

    # Filenames to be matched
    filenames_to_match = ['F0083', 'F0077', 'F0046', 'F0042', 'F0033', 'F0018', 'F0017', 'A0191', 'A0185']

    # Load images from train and test folders
    train_images = load_images(train_folder, filenames=filenames_to_match)
    test_images = load_images(test_folder, filenames=filenames_to_match)

    # Load remaining images from both datasets, excluding common images
    remaining_train_images = load_images(train_folder, exclude_filenames=filenames_to_match)
    remaining_test_images = load_images(test_folder, exclude_filenames=filenames_to_match)

    # Combine remaining train and test images
    remaining_images = {**remaining_train_images, **remaining_test_images}

    # Compute the average histogram from the remaining images
    average_reference_image = compute_average_image(remaining_images)    

    # Perform histogram matching on the common images
    matched_train_images = histogram_matching({k: v for k, v in train_images.items()}, average_reference_image)
    matched_test_images = histogram_matching({k: v for k, v in test_images.items()}, average_reference_image)

    # Save the matched images back to their original locations
    for matched_images in [matched_train_images, matched_test_images]:
        for file_id, (matched_img, filepath) in matched_images.items():
            matched_img_bgr = cv2.cvtColor(matched_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, matched_img_bgr)


def z_score_normalize_batch(batch):
    """
    Normalize a batch of grayscale images using z-score normalization.
    
    Args:
        batch (torch.Tensor): Batch of images with shape [batch_size, height, width]
    
    Returns:
        torch.Tensor: Batch of z-score normalized images
    """
    # Ensure the batch is a float tensor to avoid integer division issues
    batch = batch.float()

    # Compute mean and std for each image in the batch independently
    batch_mean = batch.mean(dim=[1, 2], keepdim=True)
    batch_std = batch.std(dim=[1, 2], keepdim=True)

    # Apply z-score normalization
    normalized_batch = (batch - batch_mean) / (batch_std + 1e-8)  # Add epsilon to avoid division by zero

    return normalized_batch



def interpolate_line(mask):
    """
    Interpolates between the start and end points of the line to make it continuous.
    
    Args:
        mask (np.ndarray): Binary mask containing a discontinuous line (0 for background, 255 for the line).

    Returns:
        np.ndarray: Mask with an interpolated continuous line.
    """
    # Find the non-zero points (the line)
    points = np.argwhere(mask > 0)

    if len(points) == 0:
        print("No line found in the mask.")
        return mask

    # Find the bounding box of the points (min/max x, y coordinates)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    # Sort points by x (or y) for interpolation
    sorted_points = points[np.argsort(points[:, 1])]

    # Draw a continuous line between points
    for i in range(1, len(sorted_points)):
        # Draw line between consecutive points
        cv2.line(mask, tuple(sorted_points[i-1][::-1]), tuple(sorted_points[i][::-1]), 255, thickness=1)

    return mask

def save_overlay_image_mask(image, mask, path):
    # Convert tensors to numpy arrays if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        # Normalize the image
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        # Remove single channel dimension if present
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]

        # Convert mask to uint8 and expand dimensions for coloring
        mask = (mask * 255).astype(np.uint8)

        # Create a 3-channel overlay where the mask will be colored (e.g., red)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert the grayscale image to BGR
        overlay[mask > 0] = [0, 0, 255]  # Color the mask area red

        # Save the resulting overlay image
        cv2.imwrite(path, overlay)


def segment_below_line(mask, line): 
    height, width = mask.shape
    below_line_mask = np.zeros_like(mask)
    y_coordinates = extract_line_y(line)
    for x in range(width):
        y = y_coordinates[x]
        if y < height:
            below_line_mask[y:,x] = mask[y:, x]
    return below_line_mask

def segment_muscle_between_lines(muscle_mask, mask, pred):
    """
    Segments the muscle between two lines using 512x512 matrix annotations.
    
    Parameters:
    - muscle_mask: A 512x512 binary mask where the muscle is segmented (1s for muscle, 0s for background).
    - line1_matrix: A 512x512 binary matrix where the first line is drawn (1s for the line, 0s elsewhere).
    - line2_matrix: A 512x512 binary matrix where the second line is drawn (1s for the line, 0s elsewhere).
    
    Returns:
    - A binary mask where the muscle between the two lines is segmented.
    """
    width, height = mask.shape
    below_line_mask = np.zeros_like(muscle_mask)
    y_coordinates = extract_line_y(mask)
    for x in range(width):
        y = y_coordinates[x]
        if y < height:
            below_line_mask[y:,x] = muscle_mask[y:, x]
    below_line_mask1 = np.zeros_like(muscle_mask)
    y_coordinates = extract_line_y(pred)
    for x in range(width):
        y = y_coordinates[x]
        if y < height:
            below_line_mask1[y:,x] = muscle_mask[y:, x]
    result = np.logical_xor(below_line_mask1, below_line_mask)
    return result.astype(int)


def calculate_percent_below(mask_supra, line_np,image):
    if isinstance(mask_supra, torch.Tensor):
        mask_supra = mask_supra.cpu().numpy()
    mask_supra = cv2.resize(mask_supra , (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)

    mask_supra = (mask_supra >0.5).astype(np.uint8)*255
    below_line_mask = segment_below_line(mask_supra, line_np)
    total_pixels = np.sum(mask_supra > 0)
    print(f"total pixels : {total_pixels}")
    below_line_pixels = np.sum(below_line_mask > 0)
    print(f"below pixels : {below_line_pixels}")
    if total_pixels == 0:
        return 0
    
    return (below_line_pixels / total_pixels) * 100 


def extract_line_y(line_np):
    height, width = line_np.shape
    y_coordinates = np.zeros(width, dtype= int)

    for x in range(width):
        column = line_np[:,x]
        non_zero_indices = np.where(column > 0)[0]
        if len(non_zero_indices) >0:
            y_coordinates[x] = non_zero_indices[0]
        else:
            y_coordinates[x] = height

    return y_coordinates

def augment_images_and_masks_in_place(image_dir, mask_dir, add_mask_dir=None):
    """
    Augments half of the images and corresponding masks with the same transformations, 
    but applies ColorJitter only to the images and saves them in the same directory.
    
    Args:
        image_dir (str): Path to the folder containing the images.
        mask_dir (str): Path to the folder containing the masks.
        add_mask_dir (str, optional): Path to the folder containing additional masks.
    """
    
    # Define a series of geometric augmentations (shared for images and masks)
    geometric_transforms = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.13, scale_limit=0.1, rotate_limit=0, p=1, border_mode=cv2.BORDER_CONSTANT)
    ])
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # If any augmented files already exist, skip
    if any(file.endswith("_aug.jpg") for file in image_files):
        return
    
    # Select half of the images randomly
    random.shuffle(image_files)
    num_to_augment = len(image_files) // 5
    selected_files = image_files[:num_to_augment]

    for image_file in selected_files:
        # Load the image and mask
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)
        
        if not os.path.exists(mask_path):
            print(f"Mask for {image_file} not found, skipping...")
            continue

        # Read image and mask as arrays
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Apply geometric transformations (identical for image and mask)
        augmented = geometric_transforms(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Process additional mask if provided
        if add_mask_dir is not None:
            add_mask_path = os.path.join(add_mask_dir, image_file)
            if os.path.exists(add_mask_path):
                mask_add = np.array(Image.open(add_mask_path).convert('L'))
                augmented_add_mask = geometric_transforms(image=image, mask=mask_add)['mask']
                Image.fromarray(augmented_add_mask).save(os.path.join(add_mask_dir, f"{os.path.splitext(image_file)[0]}_aug.jpg"))

        # Save augmented images and masks with '_aug' appended to the filename in the same folder
        Image.fromarray(augmented_image).save(os.path.join(image_dir, f"{os.path.splitext(image_file)[0]}_aug.jpg"))
        Image.fromarray(augmented_mask).save(os.path.join(mask_dir, f"{os.path.splitext(image_file)[0]}_aug.jpg"))

