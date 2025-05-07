"""
Global variables

JCA
"""
import os
from pathlib import Path

script_path, _ = os.path.split(os.path.realpath(__file__))
REPO_PATH = Path(os.path.dirname(script_path))
# REPO_PATH = str(script_path.parent.absolute())
