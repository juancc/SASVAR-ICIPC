"""
Global variables

JCA
"""
import os
from pathlib import Path

script_path, _ = os.path.split(os.path.realpath(__file__))

REPO_PATH = Path(os.path.dirname(script_path))
LABELS_PATH = os.path.join(REPO_PATH, 'Assets','SASVAR-labels.csv')
FILENAMES_PATH = os.path.join(REPO_PATH, 'Assets','app_image_filenames.pickle')
RULES_PATH = os.path.join(REPO_PATH, 'Assets','8class_v5.csv')



NAN_TAG = 'Unknown'
IMG_ACCEPTED = ["bmp", "gif", "jpeg", "png", "jpg"] # Accepted by Tensorflow
