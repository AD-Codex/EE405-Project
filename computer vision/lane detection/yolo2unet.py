import cv2
import numpy as np
import os

# Function to read YOLO segmentation labels from a text file
def read_yolo_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        yolo_labels = [list(map(float, line.strip().split())) for line in lines]
    return yolo_labels

# Function to convert YOLO segmentation labels to binary mask
def labels_to_mask(yolo_labels, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    for label in yolo_labels:
        class_index = int(label[0])
        points = [(int(label[i] * image_size[1]), int(label[i + 1] * image_size[0])) for i in range(1, len(label), 2)]
        cv2.fillPoly(mask, [np.array(points)], 255)
    return mask

# Function to save binary mask as JPEG file
def save_mask_as_jpeg(mask, output_path):
    cv2.imwrite(output_path, mask)



# Path to the YOLO segmentation label file
label_file_path = "train/labels/IMG_4983_MOV-8_jpg.rf.455e380971950b3992b23a3301d4ec27.txt"
label_dir = "train/labels"
img_dir ="train/images"
# Specify the output directory for saving JPEG masks
output_dir = "train/mask"

count=0
# Loop through each image and its YOLO label file
for filename in os.listdir(label_dir):
    
    label_path = os.path.join(label_dir, filename)
    img_pTH = os.path.join(img_dir, filename.replace(".txt", ".jpg"))
    image = cv2.imread(img_pTH)
    output_mask_path = os.path.join(output_dir, filename.replace(".txt", ".jpg"))

    # Read YOLO labels
    yolo_labels = read_yolo_labels(label_path)

    # Image size (replace with the actual size of your image)
    image_size = (640, 640)  # specify the height and width
    count+=1
    # Convert YOLO labels to binary mask
    binary_mask = labels_to_mask(yolo_labels, image_size)

    # Save binary mask as JPEG
    save_mask_as_jpeg(binary_mask, output_mask_path)
    
    # Display the binary mask
    cv2.imshow(str(count), binary_mask)
    cv2.imshow(str(filename), image)
    print(filename)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print(count)





