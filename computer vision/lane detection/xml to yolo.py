import cv2
import xml.etree.ElementTree as ET

import os

def get_file_names(directory):
    """Get all file names in the specified directory."""
    file_names = os.listdir(directory)
    return file_names

# Example usage
directory = 'D:/acadamic/fyp/datasets/sign recognition/road_signboard_detection_dataset/train'
file_names = get_file_names(directory+"/xml")
print("File names in the directory:", file_names)

class_set = set()

for i in file_names:
    xml_name=i
    jpg_name=i.split(".")[0]+".jpg"
    # print(i,jpg_name)

    # Load the XML file
    tree = ET.parse(directory+"/xml/"+ xml_name)
    root = tree.getroot()

    # Get the image path
    image_path =directory+"/jpg/"+jpg_name

    # Read the image
    image = cv2.imread(image_path)

    # Iterate through each object in the XML file
    for obj in root.findall('object'):
        # Get the object name and bounding box coordinates
        obj_name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Draw the bounding box rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
        cv2.putText(image, obj_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        class_set.add(obj_name)

    # Display the image with bounding boxes
    image=cv2.resize(image,(600,600))
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

print(class_set)
