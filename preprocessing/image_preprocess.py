import numpy as np
from tensorflow.keras.preprocessing import image as keras_image

def preprocess_xray(image_path):
    """
    Loads a chest X-ray and converts it into the (224, 224, 3) array
    the model expects, normalized to 0-1.
    """
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array