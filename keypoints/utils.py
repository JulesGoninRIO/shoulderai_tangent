from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from PIL import ImageDraw
from tqdm import tqdm
import pandas as pd


def extract_bounding_boxes_and_keypoints(image: Image.Image, model: YOLO) :
    """
    Extracts bounding boxes and keypoints from a PIL image using a YOLO model.

    Args:
        image (Image.Image): The input image in PIL format.
        model (YOLO): The YOLO model object.

    Returns:
        List[Dict]: A list of dictionaries containing bounding box coordinates and keypoints for each detection.
    """
    # Run prediction
    results = model.predict(image)

    # Extract bounding boxes and keypoints
    detections = []
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # Bounding box in [x1, y1, x2, y2] format
            keypoints = result.keypoints.xy[0].tolist()
            detections.append({"bbox": bbox, "keypoints": keypoints})
    
    return detections

def extract_tangent_sign(detection,overlayed_image=None):
    """
    Extracts the tangent sign from a detection.
    
    Args:
        detection (Dict): A dictionary containing bounding box coordinates and keypoints.
        
    Returns:
        str: The tangent sign extracted from the detection.
    """
    # Extract keypoints
    keypoints = detection["keypoints"]
    
    # Calculate the slope of the line connecting the two keypoints

    ref_keypoints = []
    if overlayed_image is not None:
        for keypoint in keypoints:
            overlayed_image.circle(keypoint,4,fill="green")
    keypoints.sort(key=lambda x: x[1])
    ref_keypoints = keypoints[:2]
    
    x1, y1 = ref_keypoints[0]
    x2, y2 = ref_keypoints[1]
    return [(x1, y1), (x2, y2)]

def draw_tangent_sign(image: Image.Image, detection):
    """
    Draws the tangent sign extracted from a detection on an image.
    
    Args:
        image (Image.Image): The input image in PIL format.
        detection (Dict): A dictionary containing bounding box coordinates and keypoints.
        
    Returns:
        Image.Image: The image with the tangent sign drawn on it.
    """
    # Extract tangent sign
    tangent_sign = extract_tangent_sign(detection)
    
    # Draw tangent sign
    draw = ImageDraw.Draw(image)
    draw.line(tangent_sign, fill="red", width=2)
    
    return image

def load_ground_truth(file_path):
    """
    Extracts the ground truth tangent sign from a file.
    
    Args:
        file_path (str): The path to the file containing the ground truth tangent sign.
        
    Returns:
        str: The ground truth tangent sign.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        keypoints = [tuple(map(int, line.strip().split(' '))) for line in lines]
    return keypoints


def get_dice(pred,gt,mask):
    below_pred = np.abs(pred - 1)
    below_gt = np.abs(gt - 1)

    intersection = np.bitwise_and(below_pred.astype(bool),below_gt.astype(bool))
    masked_intersection = np.bitwise_and(intersection.astype(bool),mask.astype(bool))

    masked_below_pred = np.bitwise_and(below_pred.astype(bool),mask.astype(bool))
    masked_below_gt = np.bitwise_and(below_gt.astype(bool),mask.astype(bool))

    a_intersect_b = masked_intersection.sum()
    a = masked_below_pred.sum()
    b = masked_below_gt.sum()
    return (2*a_intersect_b)/(a+b)

def calculate_points_above(keypoints, img_shape):
    # Extract coordinates
    x1, y1 = keypoints[0]
    x2, y2 = keypoints[1]
    
    # Calculate the slope and intercept of the line
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
    intercept = y1 - slope * x1

    # Generate a grid of x, y coordinates
    x_coords = np.arange(img_shape[0])[:, None]
    y_coords = np.arange(img_shape[1])[None, :]

    # Create a mask for points above the line
    mask = y_coords > (slope * x_coords + intercept)
    
    # Create the image with 1s above the line
    img = np.zeros(img_shape)
    img[mask] = 1

    return img.T


def perform_pred(image_path,images_out_base_path, gt_path, muscle_mask_path: str,model: YOLO):
    
    patient = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path)
    image = image.convert('RGB')
    overlayed_image = ImageDraw.Draw(image)
    detections = extract_bounding_boxes_and_keypoints(image, model)

    
    img1_mask = np.zeros(image.size)
    img2_mask = np.zeros(image.size)
    
    gt_keypoints = load_ground_truth(gt_path)

    overlayed_image.line(gt_keypoints, fill="red", width=4)
    
    img2_mask = calculate_points_above(gt_keypoints,image.size)

    if len(detections)>0:
        detection = detections[0]
        keypoints = extract_tangent_sign(detection,overlayed_image)
        overlayed_image.line(keypoints, fill="green", width=4)
        img1_mask = calculate_points_above(keypoints,image.size)
    else: # Fallback for no prediction
        img1_mask = 1 - img2_mask

    diff_mask = np.abs((img1_mask-img2_mask))

    diff = diff_mask.sum()
    score = diff/(image.size[0]*image.size[1])

    muscle_mask = np.array(Image.open(muscle_mask_path).convert('L').point(lambda x: 0 if x < 128 else 255, '1'))
    m_score = np.bitwise_and(diff_mask.astype(bool),muscle_mask.astype(bool)).sum() / muscle_mask.sum()

    dice = get_dice(img1_mask,img2_mask,muscle_mask)

    image.save(os.path.join(images_out_base_path,f"{patient}.jpg"))

    return {
        "score":score,
        "m_score":m_score,
        "diff":diff,
        "dice":dice,
        "pat":patient
    }