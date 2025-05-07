"""
Auxiliary functions

JCA
"""
import os
import Code.globals as gb

def extract_images():
    """Extract example images from repository"""
    z = os.path.join(gb.REPO_PATH, 'Dataset', 'Whatsapp30-10.zip')
    local_path =  os.path.join(gb.REPO_PATH, 'Dataset')
    os.system(f'unzip -qq "{z}" -d "{local_path}"')
    return local_path
