# import cv2

# def is_leaf(image_path):
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Perform Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours of leaf-like shapes
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Check if any contours (leaf-like shapes) are found
#     if len(contours) > 0:
#         return True
#     else:
#         return False

# # Example usage
# image_path = "human vibrant.jpg"  # Replace with the path to your image
# if is_leaf(image_path):
#     print("The image is a leaf.")
# else:
#     print("The image is not a leaf.")

import cv2
import numpy as np

def is_leaf(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper thresholds for green color (can be adjusted)
    lower_green = np.array([25, 50, 50])  # Green color range in HSV
    upper_green = np.array([80, 255, 255])

    # Threshold the image to extract green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Count the number of non-zero pixels in the mask
    num_green_pixels = np.count_nonzero(mask)

    # Calculate the percentage of green pixels relative to the total image area
    green_pixel_ratio = num_green_pixels / (image.shape[0] * image.shape[1])

    # If green pixel ratio is above a threshold, consider it as a leaf
    if green_pixel_ratio > 0.1:  # Adjust the threshold as needed
        return True
    else:
        return False

# Load the image
image = cv2.imread('bus.jpg')

# Check if the image is a leaf
if is_leaf(image):
    print("The image is a leaf.")
else:
    print("The image is not a leaf.")


# import cv2
# import numpy as np

# def is_leaf(image):
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define lower and upper thresholds for green color (adjusted values)
#     lower_green = np.array([0, 50, 50])  # Lower threshold for green color in HSV
#     upper_green = np.array([100, 255, 255])  # Upper threshold for green color in HSV

#     # Threshold the image to extract green regions
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Count the number of non-zero pixels in the mask
#     num_green_pixels = np.count_nonzero(mask)

#     # Calculate the percentage of green pixels relative to the total image area
#     green_pixel_ratio = num_green_pixels / (image.shape[0] * image.shape[1])

#     # If green pixel ratio is above a threshold, consider it as a leaf
#     if green_pixel_ratio > 0.1:  # Adjust the threshold as needed
#         return True
#     else:
#         return False
    
# import numpy as np
# import matplotlib.pyplot as plt


# test_image = cv2.imread('human face.jpg')

# # Check if the image is a leaf
# if is_leaf(test_image):
#     print("The image is a leaf.")

    
# else:
#     print("The image is not a leaf.")
#     print("Enter the image of a leaf")
    
        
