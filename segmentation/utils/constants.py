# Directory paths
ORIGINAL_DATA_DIR = "/data/soin/shoulder_ai/data/2024/DATA"

# Model parameters
PLANE_TYPES = ["sagittal", "coronal"]
PHASES = ["train", "test"]
SLICES = ["all_slices", "ROI_slice"]
SHAPE = (512, 512)

# Image types
SAG_IMAGE_TYPES = [
    "T1_sag",
    "T1_in_phase_sag",
    "T2_fs_sag",
    "T1_fat_sag",
    "T1_water_sag",
    "T2_sag_spair"
]
SAG_IMAGE_MAIN_TYPES = ["T1_sag", "T1_in_phase_sag"]
SAG_IMAGE_ADDITIONAL_TYPES = ["T2_fs_sag", "T1_fat_sag", "T1_water_sag", "T2_sag_spair"]
SAG_IMAGE_MASK_TYPES = ["T1_sag", "T1_in_phase_sag", "T1_fat_sag", "T2_sag_spair"]

COR_IMAGE_TYPES = ["T1_fs_cor", "T2_fs_cor"]
COR_IMAGE_MAIN_TYPES = ["T2_fs_cor"]
COR_IMAGE_ADDITIONAL_TYPES = ["T1_fs_cor"]

# Semantic segmentation masks
MASKS = {
    "sagittal": [
        "tangent_sign",
        "infraspinatus",
        "supraspinatus",
        "subscapular",
        "teres_minor",
    ],
    "coronal": ["best_fit_circle"],  # TODO add triangle shaped segmentation?
}

# Mean and standard value accross all train dataset for each image type
MEAN_STD_ROI_SAG = {
    "T1_sag" : (0.17913025617599487, 0.17865481972694397),
    "T2_fs_sag" : (0.07140810787677765, 0.05540060997009277),
    "T1_water_sag" : (0.1197919026017189, 0.1021985188126564),
    "T1_fat_sag" : (0.12768253684043884, 0.17902158200740814),
    "T1_in_phase_sag" : (0.17601440846920013, 0.16876275837421417)
}


MEAN_STD_ALL_SAG = {
    "T1_sag" : (0.17913025617599487, 0.17865481972694397),
    "T1_in_phase_sag" : (0.17601440846920013, 0.16876275837421417)
}

MEAN_STD_ALL_COR = {}  # TODO
MEAN_STD_ROI_COR = {}  # TODO
