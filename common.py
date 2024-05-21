import os
import cv2
import sys

def list_image_path(data_path):
    """
    Input: data image paths
    Return: list path images
    """
    images_path = []
    if os.path.exists(data_path):
        for file_name in os.listdir(data_path):
            if file_name.endswith('.png'):
                image_path = os.path.join(data_path, file_name)
                images_path.append(image_path)
    else:
        print("data path not found")
        sys.exit(1)
    return images_path

