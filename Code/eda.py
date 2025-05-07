"""
Exploratory Data Analysis

JCA
"""

import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import Code.globals as globals


def fix_count(count, ids, target_classes=None):
    """Count the number of images by ids"""
    if target_classes:
        fix_count = {name: 0 for name in target_classes}
    else:
        fix_count = {name: 0 for idx, name in ids.items()}


    for i,no in count.items():
        # if i>=0:
        label = ids[i]
        if target_classes:
            if not label in target_classes:
                continue
        fix_count[label] = int(no)
        # else:
        #     fix_count[NAN_TAG] = int(no)
    return fix_count


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


def bar_plot(labels, column_ids, category, bottom=None, label=None, target_classes=None, color=None):
    """Bar plot of a category of ONE dataset. Used by stacket_bar_plot"""
    barcolor = color if color else [random.random() for r in 'RGB']
    ids = column_ids[category]
    labels_dataset = pandas.DataFrame(data=labels)

    count = dict(labels_dataset[category].value_counts())
    count = fix_count(count, ids, target_classes=target_classes)
    data = list(count.values())
    labels = list(count.keys())
    bar_container = plt.bar(labels, data,  color=barcolor, bottom=bottom, label=label)
    return np.array(data), bar_container, labels


def stacket_bar_plot(d_labels, column_ids, category, title_size=15, target_classes=None, colors=None, figsize=(16,8)):
    # plt.figure(figsize = (16,8))
    fig, ax = plt.subplots(figsize=figsize)

    bottom = None
    labels_plot = []
    for name, labels in d_labels.items():
        # labels = [l.capitalize() for l in labels]
        color  = colors[name] if colors else None
        if bottom is None:
            bottom, bar_container, labels_plot = bar_plot(labels, column_ids, category, bottom=bottom, label=name, target_classes=target_classes, color=color)
        else:
            bot, bar_container, labels_plot = bar_plot(labels, column_ids, category, bottom=bottom, label=name, target_classes=target_classes, color=color)
            bottom += bot

    plt.title(category, fontsize=title_size)
    # plt.xticks(rotation=90, fontsize=10)
    plt.xticks([]) # dont show x labels under the plot
    # Plot labels above bars
    ax.bar_label(bar_container, labels=labels_plot, rotation=90, fontsize=10, padding=8)
    # Let some space for the label with more images
    ax.set_ylim(top=max(bottom)+len(labels_plot[0])*40)  

    plt.legend()
    plt.show()

    return bottom


def dataset_source_plot(d_labels, colors):
    """Plot distribution of images by dataset source. Return material count"""
    d_count = {name:len(labels['Material']) for name, labels in d_labels.items()}
    labels = list(d_count.keys())
    data = list(d_count.values())

    labels = [x for _, x in sorted(zip(data,labels), reverse=True )]
    data = sorted(data, reverse=True )

    plt.bar(labels, data, color=[colors[l] for l in labels])
    plt.title('Dataset Sources')
    plt.show()

    # return total
    return np.sum(data)

def correlate_plot(labels, column_ids, column0, column1, show_legend=True, figsize=(16,8)):
    """Bar plot of one label distribution vs other"""
    x_data = np.array(labels[column0])
    y_data = np.array(labels[column1])

    x = list(column_ids[column0].keys())
    y = list(column_ids[column1].keys())
    x_plot = [column_ids[column0][i] for i in x]

    bottom = np.zeros(len(x_plot))
    # plt.figure(figsize = (16,8))
    fig, ax = plt.subplots(figsize = figsize)
    
    for i in y:
        sl_x = x_data[y_data==i]
        y_plot = []
        x_count = Counter(sl_x)
        
        for j in x:
            y_plot.append(x_count[j])
        bar_container = plt.bar(x, y_plot,  color=[random.random() for r in 'RGB'], bottom=bottom, label=column_ids[column1][i])


        bottom = bottom + np.array(y_plot)
    
    # Plot labels above bars
    ax.bar_label(bar_container, labels =x_plot, rotation=90, fontsize=10, padding=8)
    
    plt.title(f'{column0} vs. {column1}')
    # plt.xticks(rotation=90, fontsize=10)
    plt.xticks([]) # dont show x labels under the plot

    # Let some space for the label with more images
    ax.set_ylim(top=max(bottom)+ len(x_plot[0])*40)  
    if show_legend: plt.legend(fontsize=8)
    plt.show()
            

# def capacity_plot(labels, column_ids, column0, show_legend=True):
#     """Plot capacity vs other column"""
#     pass

def pie(column_ids, labels, target_column, src_column, src_label):
    """Pie plot distribution of target_label by column0"""
    # Search id by label
    label_id = None
    for k,v in column_ids[src_column].items():
        if v==src_label: 
            label_id = k
            break
    if not label_id:
        print(f'Label id for {src_label} not found')
        return

    src_labels = np.array(labels[src_column])
    target_labels = np.array(labels[target_column])
    
    dist = target_labels[src_labels==label_id]

    count = dict(Counter(dist))
    dist = []
    labels = []
    for k,v in count.items():
        dist.append(v)
        if k >=0:
            labels.append(column_ids[target_column][k])
        else:
            labels.append(gb.NAN_TAG)
    plt.pie(dist, labels=labels)
    plt.title(f'{target_column} in {src_label}') # column_ids[src_column][label_id]

    plt.show() 




def plot_labels_dist(dataset, id_labels, color='royalblue', figsize=(12,5), fontsize=10):
    """Plot distribution of dataset labels
        Parameters:
            :dataset (np.array): Features matrix that its last column is the ordinal labels. 
                Returned by load_features function
            :id_labels (dict): labels id names
    """
    y = dataset[:,-1]
    dist = dict(Counter(y))
    labels = [id_labels[int(i)] for i in dist.keys()]
    fig = plt.figure(figsize=figsize)
    
    plt.bar(labels, list(dist.values()),  color=color)
    plt.xticks(fontsize=fontsize)
    plt.title('Dataset class distribution')

    plt.show()
    out = {id_labels[int(i)]:j for i,j in dist.items()}
    return out


def plot_count(dt_count, id_labels, color='royalblue', figsize=(12,5), fontsize=10):
    """Plot distribution of dataset by its count
        Parameters:
            :dt_count (collections.Counter): Ordered dict of {label:count}
            :id_labels (dict): labels id names
    """
    dist = dict(dt_count)
    labels = [id_labels[int(i)] for i in dist.keys()]
    fig = plt.figure(figsize=figsize)
    
    plt.bar(labels, list(dist.values()),  color=color)
    plt.xticks(fontsize=fontsize)
    plt.title('Dataset class distribution')

    plt.show()
    out = {id_labels[int(i)]:j for i,j in dist.items()}
    return out




def plot_samples(dataset, id_labels, figsize=(2,4), scale=2, font_size=9):
    """Plot images from Tf dataset"""
    show_batch = next(iter(dataset))

    total_samples = np.prod(figsize)
    numpy_images = show_batch[0].numpy()[:total_samples]
    numpy_labels = show_batch[1].numpy()[:total_samples]

    # Plot images
    i=1
    fig = plt.figure(figsize=figsize)
    fig.set_size_inches(figsize[1]*scale, figsize[0]*scale)
    for im, enc_label in zip(numpy_images, numpy_labels):
        ax = fig.add_subplot(figsize[0], figsize[1], i)

        label = id_labels[np.argmax(enc_label)]

        ax.set_title(f'{label}', fontsize=font_size)

        plt.imshow(im/255)
        plt.axis('off')

        i+=1
    
    plt.show()



def showim(im, scale=0.5, window='Image'):
    """Show image using cv2"""
    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def popular_samples(d_labels, column_ids, category, no=50):
    """Get the labels by category of the no popular by amount of images """
    ids = column_ids[category]
    data_count = []
    labels_out = []
    for name, labels in d_labels.items():
        labels_dataset = pandas.DataFrame(data=labels)

        count = dict(labels_dataset[category].value_counts())
        count = fix_count(count, ids)

        data = list(count.values())
        l = list(count.keys())
        if len(l) > len(labels_out): labels_out = l
        data_count.append(data)
    
    data_count = np.array(data_count).sum(axis=0)
    labels_sorted = [x for _, x in sorted(zip(data_count,labels_out), reverse=True )]
    return labels_sorted[:no]



def plot_distribution(labels, column_ids, category, no=100):
    """Shorter function to plot dataset distribution by category"""
    popular_y = popular_samples({'App':labels}, column_ids, category, no=no)
    y_count = stacket_bar_plot({'App':labels}, column_ids, category, target_classes=popular_y, colors=None)