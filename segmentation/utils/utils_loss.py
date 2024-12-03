import cv2
import numpy as np
import torch
from utils.constants import SHAPE
import torch.nn.functional as F
import math
import os 


def get_slope(pt0, pt1):
    """Getter function for slope of line defined by two points.

    Args:
        pt0 (ndarray): X, Y coordinates of point 0.
        pt1 (ndarray): X, Y coordinates of point 1.

    Returns:
        float: Slope of the line.

    """
    if isinstance(pt0, np.ndarray):
        pt0 = torch.tensor(pt0)
        pt0 = pt0.clone().detach()
    if isinstance(pt1, np.ndarray):
        pt1 = torch.tensor(pt1)
        pt1 = pt1.clone().detach()
    x0, y0 = pt0.unbind(-1)
    x1, y1 = pt1.unbind(-1)

    return (y1 - y0) / (x1 - x0)


def get_slope_list(pt0, pt1):
    if pt1[0] - pt0[0] != 0:
        return (pt1[1] - pt0[1]) / (pt1[0] - pt0[0])
    else:
        return float("inf")

def calculate_lenght(pt0, pt1):
    if not isinstance(pt0, torch.Tensor):
        pt0 = torch.tensor(pt0, dtype = torch.float32)

    if not isinstance(pt1, torch.Tensor):
        pt1 = torch.tensor(pt1, dtype = torch.float32)

    
    lemght = torch.norm(pt0-pt1)
    return lemght





def get_angle_to_horizontal(pt0, pt1):
    """Getter function for angle between line defined by two points
    and the horizontal line.

    Args:
        pt0 (ndarray): X, Y coordinates of point 0.
        pt1 (ndarray): X, Y coordinates of point 1.

    Returns:
        float: Angle between line and horizontal line.
    """
    # Find angle between line and the horizontal line.
    pt0 = torch.tensor(pt0, dtype=torch.float32)
    pt1 = torch.tensor(pt1, dtype=torch.float32)
    
    slope = get_slope(pt0, pt1)
    
    angle = torch.atan(slope)
    angle = torch.rad2deg(angle)

    return angle


def prolong_line(pt0, pt1, img_shape):
    slope = get_slope(pt0, pt1)
    intercept = pt0[1] - slope*pt0[0]
    x_min , x_max = 0, img_shape[1]
    y_min = slope*x_min + intercept
    y_max = slope*x_max + intercept

    pt0_prolo = torch.tensor([x_min, y_min], dtype = torch.float32)
    pt0_prolo = pt0_prolo.clone().detach()
    pt1_prolo = torch.tensor([x_max, y_max], dtype = torch.float32)
    pt1_prolo = pt1_prolo.clone().detach()

    return pt0_prolo , pt1_prolo


def calculate_mse(target_slope, target_intercept, pred_slope, pred_intercept, img_shape):
    
    x_values = torch.arange(0, img_shape[1], dtype = torch.float32)
    target_y = target_slope*x_values + target_intercept
    pred_y = pred_slope*x_values + pred_intercept
    mse = torch.mean((target_y - pred_y)**2)

    return mse.item()

    
def find_intersection(pt0, pt1, pt2, pt3):
    x0, y0 = pt0
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 =pt3

    m1 = (y1 - y0) / ( x1- x0)
    c1 = y0 - m1 * x0
    m2 = (y3 - y2) / (x3-x2)
    c2 = y2 - m2*x2
    x_inter = (c2-c1) / (m1-m2)
    y_inter = m1*x_inter + c1
    inter = torch.tensor([x_inter, y_inter], dtype =torch.float32)
    return inter.clone().detach()

def calculate_angle(pt0, pt1, pt2, pt3, img_shape):
    pt0_prolo , pt1_prolo = prolong_line(pt0, pt1, img_shape)
    pt2_prolo, pt3_prolo = prolong_line(pt2, pt3, img_shape)

    inter = find_intersection(pt0_prolo,pt1_prolo,pt2_prolo,pt3_prolo)

    angle1 = get_angle_to_horizontal(pt0_prolo, inter)
    angle2 = get_angle_to_horizontal(pt2_prolo, inter)

    angle_line = torch.abs(angle1 - angle2)
    return angle_line



def write_infinite_line(pt0, pt1, img):
    """Function to write infinite line on image from two points.

    Args:
        pt0 (ndarray): X, Y coordinates of point 0.
        pt1 (ndarray): X, Y coordinates of point 1.
        img (ndarray): Black and white image.

    Returns:
        ndarray: Black and white image with drawn infinite line.
    """
    pt0 = list(pt0)
    pt1 = list(pt1)

    p = [0, 0]
    q = [0,0]


    slope = get_slope_list(pt0, pt1)
    p[1] = -(pt0[0] - p[0]) * slope + pt0[1]
    q[1] = -(pt1[0] - q[0]) * slope + pt1[1]

    p[0]= 0
    q[0] = SHAPE[1]

    p[1] = int(p[1])
    q[1] = int(q[1])

    cv2.line(img, tuple(p), tuple(q), 255, 1)
    return img


def get_line_end_points(img, max_pixel_val):
    """Getter function for line end points on a given image.

    Args:
        img (ndarray): Black and white image considered.
        max_pixel_val (int, optional): Maximum pixel value. Defaults to 255.

    Returns:
        tuple: tuple of end points.
    """
    # Read image pixels from the left, column by column until we find a white pixel
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

   
    if img.ndim == 3:
        img =img[0,:,:]
    pt0 = []
    for x in range(SHAPE[1]):
        for y in range(SHAPE[0]):
            if img[y][x] == max_pixel_val:
                pt0 = [x, y]
                break
        if len(pt0) >0:
            break
    # Read image pixels from the right, column by column until we find a white pixel
    pt1 = []
    for x in reversed(range(SHAPE[1])):
        for y in reversed(range(SHAPE[0])):
            if img[y][x] == max_pixel_val:
                pt1 = [x, y]
                break
        if len(pt1) >0:
            break
       
    return np.array(pt0), np.array(pt1)

def get_end_points_mask(mask):
    "Get end points of already extended mask"
    points = []
    first_row = np.nonzero(mask[0])[0]
    if len(first_row) != 0:
        points.append([0, first_row[math.ceil(len(first_row)/2)]])
    last_row = np.nonzero(mask[-1])[0]
    if len(last_row) != 0:
        points.append([mask.shape[0],last_row[math.ceil(len(last_row)/2)]])
    if len(points) == 2:
        return torch.tensor(points)
    first_column = np.nonzero(mask[:, 0])[0]
    if len(first_column) != 0:
        points.append([first_column[math.ceil(len(first_column)/2)],0])
    if len(points) == 2:
        return torch.tensor(points)
    last_column = np.nonzero(mask[:, -1])[0]
    if len(last_column) != 0:
        points.append([last_column[math.ceil(len(last_column)/2)], mask.shape[1]])
    return torch.tensor(points)


def get_line_end_points_pred(img, threshold = 0.5):
    """Getter function for line end points on a given image.

    Args:
        img (ndarray): Black and white image considered.
        max_pixel_val (int, optional): Maximum pixel value. Defaults to 255.

    Returns:
        tuple: tuple of end points.
    """
    # Read image pixels from the left, column by column until we find a white pixel
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    if img.ndim == 3:
        img =img[0,:,:]
    pt0 = []
    for x in range(SHAPE[1]):
        for y in range(SHAPE[0]):
            if img[y][x] >= threshold:
                pt0 = [x, y]
                break
        if len(pt0) >0:
            break
        

    # Read image pixels from the right, column by column until we find a white pixel
    pt1 = []
    for x in reversed(range(SHAPE[1])):
        for y in reversed(range(SHAPE[0])):
            if img[y][x] >= threshold:
                pt1= [x, y]
                break
        if len(pt1) >0:
            break
        
    return np.array(pt0), np.array(pt1)


def get_lines_height_diff(line0_y0, line0_y1, line1_y0, line1_y1):
    """Getter function for height differenc of two lines on an image.

    Args:
        line0_y0 (int): Y coordinate of point 0 of line 0.
        line0_y1 (int): Y coordinate of point 1 of line 0.
        line1_y0 (int): Y coordinate of point 0 of line 1.
        line1_y1 (int): Y coordinate of point 1 of line 1.

    Returns:
        int: line height difference in pixels
    """
    # Compare middle of lines: compute line middle points, compare heights
    line0_middle_y = (line0_y0 + line0_y1) / 2
    line1_middle_y = (line1_y0 + line1_y1) / 2

    # TODO Alternate method:
    # Compare middle of image (x=256): write infinite line, compare heights
    
    return torch.abs(line0_middle_y - line1_middle_y)

def compare_lines_mid_height(img_shape, target_line, pred_line):
    """
    Compare the heights of two lines at the middle of the image and return the difference.
    
    Args:
        img_shape (tuple): Shape of the image as (height, width).
        line0_y0 (torch.Tensor): y-coordinate of the start point of line 0.
        line1_y0 (torch.Tensor): y-coordinate of the start point of line 1.

    Returns:
        torch.Tensor: The absolute difference in height between the two lines at the image middle.
    """
    img_center_x = img_shape[1] // 2
    
    def y_at_x(slope, intercept, x):
        y = slope*x + intercept
        return y

    # Calculate the y-intercepts of each line at the middle of the image using tensors
    target_slope, target_intercept = target_line
    pred_slope, pred_intercept = pred_line

    target_y = y_at_x(target_slope, target_intercept, img_center_x)
    pred_y = y_at_x(pred_slope, pred_intercept, img_center_x)

    # Calculate the difference in height between the two lines
    height_difference = torch.abs(target_y - pred_y)

    return height_difference

def get_y_intercept_at_x(slope, x):
    """
    Calculate the y-intercept of a line defined by two points at a given x coordinate.

    Args:
        pt0 (tuple): The first point (x, y) on the line.
        pt1 (tuple): The second point (x, y) on the line.
        x (int): The x coordinate at which to find the y-intercept.

    Returns:
        int: The y-intercept of the line at the given x coordinate.
    """
    
    y_intercept = pt0[1] - (slope * (pt0[0] - x))

    return y_intercept

def pixel_count(pred, target):
    pred_count = np.sum(pred)
    target_count = np.sum(target)
    pixel_count_loss = torch.tensor((pred_count - target_count)**2, device = pred.device, dtype = torch.float32)
    return pixel_count_loss


def calculate_linearity(line):
    """
    Calculate the linearity of a given line.

    Args:
        line (numpy.ndarray): An array of points representing the line.

    Returns:
        float: The linearity measure of the line.
    """
    line_np = line.cpu().detach().numpy()
    # Ensure the input is a numpy array
    line = np.array(line_np)
    
    # Calculate the differences between consecutive points
    diffs = np.diff(line, axis=0)
    
    # Calculate the angles of the segments
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    
    # Calculate the differences between consecutive angles
    angle_diffs = np.diff(angles)
    
    # Calculate the variance of the angle differences
    linearity = np.var(angle_diffs)
    
    return linearity


def get_intercept(pt0, slope):
    return pt0[1] - slope*pt0[1]

def add_loss_the_foldername(folder, loss):
    """
    Adds the loss value to the beginning of the folder name.

    Parameters:
    folder (str): The path to the folder.
    loss (float): The loss value to add to the beginning of the folder name.

    Returns:
    str: The new folder name with the loss value prefixed.
    """
    # Get the absolute path of the folder
    folder_path = os.path.abspath(folder)
    
    # Get the parent directory and the current folder name
    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    
    # Format the loss value and create the new folder name
    new_folder_name = f"{loss:.4f}_{folder_name}"  # Adjust the precision of loss as needed
    
    # Combine the parent directory with the new folder name
    new_folder_path = os.path.join(parent_dir, new_folder_name)
    
    # Rename the folder
    os.rename(folder_path, new_folder_path)
    
    return new_folder_path
