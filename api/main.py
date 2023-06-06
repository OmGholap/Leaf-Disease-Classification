from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("C:/Users/vishi/Documents/RND-Drip/saved_models/2")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


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






@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Check if the image is a leaf
    if is_leaf(image):
        predicted_class = "Leaf"
        confidence = 1.0
        is_plant = True

        # Make predictions using the loaded model
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
        'class': predicted_class,
        'confidence': float(confidence),
        'is_plant': is_plant
        }
    else:
        is_plant = False
        print("enter the image of a plant")
        return {
        'is_plant': is_plant,
        'image': "Enter the image of a plant "
        }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
