"""
Exploratory Data Analysis

JCA
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import Code.globals as globals

def plot_random_images(n=10, img_exts={'.jpg', '.jpeg', '.png', '.bmp', '.gif'}):
    base_path = os.path.join(globals.REPO_PATH, 'Dataset')
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_exts and not file.startswith('.'):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        print("No images found.")
        return

    # Select random images
    selected_paths = random.sample(image_paths, min(n, len(image_paths)))

    # Determine grid size
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    # Plot mosaic
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axs = axs.flatten() if n > 1 else [axs]

    for ax, img_path in zip(axs, selected_paths):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path), fontsize=6)
        ax.axis('off')

    # Hide unused subplots
    for ax in axs[len(selected_paths):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
