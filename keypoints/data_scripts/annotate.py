import cv2
import os
import json

# Global variables
rectangle = []
points = []
current_image = None
original_image = None
image_name = None
annotations = {}




def main():
    global current_image, original_image, image_name, rectangle, points, annotations

    directory = "/data/soin/shoulder_ai/src/tangent sign/data/keypoints/original"
    
    # Load existing annotations
    images_directory = os.path.join(directory,"images")
    annotations_doc_directory = os.path.join(directory,"annotations","txt")
    annotations_yolo_directory = os.path.join(directory,"annotations","yolo")
    # Get list of images in the directory
    images = [img for img in os.listdir(images_directory) if img.endswith((".png", ".jpg", ".jpeg"))]
    annotations = load_annotations(annotations_doc_directory)

    # Loop through each image
    for img_name in images:
        image_name = img_name

        # Skip image if already annotated
        if image_name in annotations:
            print(f"Skipping {image_name}, already annotated.")
            continue

        image_path = os.path.join(images_directory, img_name)
        annotation_path = os.path.join(annotations_doc_directory,image_name.replace(".jpg",".txt"))
        original_image = overlay_annotation(image_path,annotation_path)
        if original_image is None:
            print(f"Unable to read {image_name}")
            continue
        img_size = original_image.shape
        print(img_size)
        # Reset points and rectangle for the new image
        reset_view()

        cv2.setMouseCallback("Image", draw)

        # Wait for key press to save, reset, or quit
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to save
                if len(rectangle) == 2 and len(points) == 3:
                    save_annotations(image_name,annotations_yolo_directory, img_size, rectangle, points)
                    print(f"Annotations saved for {image_name}")
                    break
                else:
                    print("Please complete the annotation (draw a rectangle and select 3 points).")
            elif key == ord('r'):  # Press 'r' to reset
                print(f"Resetting view for {image_name}.")
                reset_view()
            elif key == ord('q'):  # Press 'q' to quit
                print("Exiting...")
                return

    cv2.destroyAllWindows()


def annotation_to_yolo(annotation,img_size):
    img_size_x,img_size_y, _ = img_size
    rectangle = annotation["rectangle"]
    rec_c0_x,rec_c0_y = rectangle[0]
    rec_c1_x,rec_c1_y = rectangle[1]
    
    bbox_x = (float(rec_c0_x+rec_c1_x)/2)
    bbox_y = (float(rec_c0_y+rec_c1_y)/2)
    bbox_w = abs(rec_c0_x-rec_c1_x)
    bbox_h = abs(rec_c0_y-rec_c1_y)

    
    points = annotation["points"]
    p1_x,p1_y = points[0]
    p2_x,p2_y = points[1]
    p3_x,p3_y = points[2]

    #normalize to 0-1
    bbox_x = float(bbox_x) / img_size_x
    bbox_y = float(bbox_y) / img_size_y
    bbox_w = float(bbox_w) / img_size_x
    bbox_h = float(bbox_h) / img_size_y

    p1_x = float(p1_x) / img_size_x
    p1_y = float(p1_y) / img_size_y
    
    p2_x = float(p2_x) / img_size_x
    p2_y = float(p2_y) / img_size_y

    p3_x = float(p3_x) / img_size_x
    p3_y = float(p3_y) / img_size_y



    out_string = f"0 {bbox_x} {bbox_y} {bbox_w} {bbox_h} {p1_x} {p1_y} {p2_x} {p2_y} {p3_x} {p3_y}"
    return out_string



# Mouse callback function
def draw(event, x, y, flags, param):
    global rectangle, points, current_image

    # Left mouse button down: start rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(rectangle) < 2:
            rectangle = [(x, y)]  # Start point of the rectangle
        elif len(points) < 3:
            points.append((x, y))  # Collect up to 3 points

    # Mouse move: update rectangle in real-time
    elif event == cv2.EVENT_MOUSEMOVE:
        if len(rectangle) == 1:
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, rectangle[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)

    # Left mouse button up: finalize rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        if len(rectangle) == 1:
            rectangle.append((x, y))  # End point of the rectangle
            cv2.rectangle(current_image, rectangle[0], rectangle[1], (0, 255, 0), 2)
            cv2.imshow("Image", current_image)

    # Draw points on the image when clicked
    if len(points) > 0:
        for point in points:
            cv2.circle(current_image, point, 5, (0, 0, 255), -1)
        cv2.imshow("Image", current_image)

# Load existing annotations from JSON file
def load_annotations(annotations_path):
    if os.path.exists(annotations_path):
        files = os.listdir(annotations_path)
        return {f:True for f in files if f.endswith(".txt")}
    return {}

# Save annotation to txt file in yolo format
def save_annotations(image_name,annotations_dir, img_size , rectangle, points):
    annotations[image_name] = True
    annotation = {
        "rectangle": rectangle,
        "points": points
    }
    out_string = annotation_to_yolo(annotation,img_size)
    if not os.path.exists(annotations_dir):
        os.os.makedirs(annotations_dir)
    with open(os.path.join(annotations_dir,image_name.replace(".jpg",".txt")),"w") as f:
        f.write(out_string)

# Reset the view to the original image and clear annotations
def reset_view():
    global current_image, rectangle, points
    current_image = original_image.copy()
    rectangle = []
    points = []
    cv2.imshow("Image", current_image)

def overlay_annotation(image_path,annotation_path):

    image = cv2.imread(image_path)
    if image is None:
        return None
    with open(annotation_path, 'r') as file:
        points = file.readlines()

    # Parse the points
    x1, y1 = map(int, points[0].split())
    x2, y2 = map(int, points[1].split())

    # Calculate the slope (m) and intercept (c) of the line y = mx + c
    if x2 != x1:  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        # If x1 == x2, the line is vertical
        slope = None

    # Determine the line endpoints across the whole image
    height, width = image.shape[:2]

    if slope is not None:
        # For non-vertical lines, find y at x=0 and x=width-1
        y_start = int(slope * 0 + intercept)
        y_end = int(slope * (width - 1) + intercept)
        start_point = (0, y_start)
        end_point = (width - 1, y_end)
    else:
        # For vertical lines, extend through the height of the image
        start_point = (x1, 0)
        end_point = (x1, height - 1)

    # Draw the line on the image
    line_color = (255, 0, 0)  # Blue
    line_thickness = 2
    cv2.line(image, start_point, end_point, line_color, line_thickness)
    return image


    


if __name__ == "__main__":
    main()

