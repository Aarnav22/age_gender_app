import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img):
    """
    Preprocess input image to feed into model
    - Resize to match model input shape
    - Normalize pixel values
    - Expand dims for batch
    """
    img = cv2.resize(img, (64, 64))  # Adjust if your model uses different size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img
