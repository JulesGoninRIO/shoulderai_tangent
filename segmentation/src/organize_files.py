import os
import shutil
import re
import pandas as pd
import numpy as np
import csv

import cv2
import pandas as pd
import shutil
import torch
mask_folder = "./data/sagittal/ROI_slice/test/masks/T1_sag/supraspinatus/"
mask_folder_t = "./data/sagittal/ROI_slice/test/masks/T1_sag/tangent_sign/"


def load_mask_supra(patient_name):    
    mask_path = os.path.join(mask_folder,f"{patient_name}.jpg")
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img = (mask_img > 127).astype(np.uint8)*255
    else:
        print(f"Mask for patient {patient_name} not found")
        print(mask_path)
        mask_img = None

    return mask_img

def load_mask_tangent(patient_name):
    mask_path = os.path.join(mask_folder_t,f"{patient_name}.jpg")
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img = (mask_img > 127).astype(np.uint8)*255
    else:
        print(f"Mask for patient {patient_name} not found")
        mask_img = None

    return mask_img

    
def delete_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete")

def save_array_to_csv(array, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each element of the array in a new row
        for item in array:
            writer.writerow([item])



def compute_average_loss(folder_path):
    # Set the directory for the output CSV file
    merged_file_path = os.path.join(folder_path, 'merged_numbers.csv')
    print(merged_file_path)

    # Initialize empty lists to store Dice coefficients and differences
    patient_ids = []
    dice_coeffs = []
    differences = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .csv file
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file
            # try:
                # Read the CSV file expecting two columns
            data = pd.read_csv(file_path, header=None)  # No header row
                # Append Dice coefficients and differences to their respective lists
            patient_ids.extend(data.iloc[0].tolist())
            if len(data)>1:
                dice_coeffs.extend(data.iloc[1].tolist())
                differences.extend(data.iloc[2].tolist())
            # except Exception as e:
            #     print(f"Warning: Could not read {filename}. Error: {e}")
            
            # Delete the original file after reading
            # os.remove(file_path)
    # Convert values to floats (if necessary)
    dice_coeffs = [float(num) for num in dice_coeffs]
    differences = np.array([float(num) for num in differences])
    # Write all numbers to a new merged CSV file with two columns
    merged_data = pd.DataFrame({
        "Patient id": patient_ids, 
        'Dice Coeff': dice_coeffs,
        'Difference': differences
    })
    merged_data.to_csv(merged_file_path, index=False)
    differences = differences[np.isfinite(differences)]
    dice_coeffs = np.array(dice_coeffs)
    # Calculate and print mean and standard deviation for Dice coefficients
    if len(dice_coeffs) > 0:  # Check if the list is not empty
        print("Mean Dice Coeff:", np.mean(dice_coeffs))
        print("Standard Deviation:", np.std(dice_coeffs))
        print("Percent where dice is smaller than 5%", np.sum(dice_coeffs >= 0.95)/len(dice_coeffs)*100)
        print("Percent where dice is smaller than 10%", np.sum(dice_coeffs >= 0.9)/len(dice_coeffs)*100)
        print("Percent where dice is smaller than 15%", np.sum(dice_coeffs >= 0.85)/len(dice_coeffs)*100)
        print("Percent where dice is smaller than 20%", np.sum(dice_coeffs >= 0.8)/len(dice_coeffs)*100)
    else:
        print("No Dice coefficients were found to compute mean and std.")
    print("Percent where diff is smaller than 5%", np.sum(differences <= 5)/len(differences)*100)
    print("Percent where diff is smaller than 10%", np.sum(differences <= 10)/len(differences)*100)
    print("Percent where diff is smaller than 15%", np.sum(differences <= 15)/len(differences)*100)
    print("Percent where diff is smaller than 20%", np.sum(differences <= 20)/len(differences)*100)
    print("Average diff", np.nanmean(differences))
    print("Std of difference", np.nanstd(differences))




