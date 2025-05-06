"""
Auxiliary functions

JCA
"""
import os

def extract_images(repo_path):
    """Extract example images from repository"""
    z = os.path.join(repo_path, 'Dataset', 'Whatsapp30-10.zip')
    local_path =  os.path.join(repo_path, 'Dataset')
    os.system(f'unzip -qq "{z}" -d "{local_path}"')
    return local_path
