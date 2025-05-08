"""
Model functions

JCA
"""
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np



def load_image(img_path, target_size=(224, 224), preprocess_input=preprocess_input):
    """Read and preprocess image for ResNet50"""
    img = image.load_img(img_path, target_size=target_size)  # ResNet50 expects 224x224
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Preprocess for ResNet50
    return x