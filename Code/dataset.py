"""
Dataset related function

JCA
"""
import pickle
import os
import random

import pandas
import numpy as np


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pandas.set_option('display.width', 200)  # or some large value
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_colwidth', None)

import Code.globals as gb




def load_labels(materials_replace=None):
    """Load CSV with SASVAR labels"""
    # Load labels

    # Target columns
    column_ids = {'Material':{}, 'Brand':{}, 'Type':{},
                  'Dirt':{}, 'Package Color':{}, 'Bottle Cap':{},
                  'ProductType':{}}

    df = pandas.read_csv(gb.LABELS_PATH, sep=',')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.drop(columns=['Capacity', 'Reference'],
            inplace=True, errors='ignore') # 'Nombre producto'

    df.rename(columns = {'Packaging':'Type'}, inplace=True)
    df.rename(columns={'Nombre producto': 'Name'}, inplace=True)
    df.rename(columns={'Botella cap': 'Bottle Cap'}, inplace=True)
    df.rename(columns={'Product category': 'ProductType'}, inplace=True)

    # Replace Verdadero Falso for True, False
    df.replace({'VERDADERO': 'TRUE', 'FALSO': 'FALSE', 'VERDADERO ': 'TRUE'}, inplace=True)

    # Replace material names
    if materials_replace:
        df.replace({
        'Material': materials_replace,
        }, inplace=True)

    for col in column_ids.keys():
        # Get unique labels
        uniques = list(df[col].unique())
        # # Remove nan from id dicts
        if np.nan in uniques: uniques.remove(np.nan)
        ids = dict(enumerate(uniques))
        ids[-1] = gb.NAN_TAG

        # Reverse IDs for table replacement
        df.replace({col: {v:k for k,v in ids.items()}}, inplace=True)
        column_ids[col] = ids

    # NAN as -1 id
    df.fillna(-1, inplace=True)

    return df, column_ids


def calculate_new_label(row, column_ids, rules):
    """Takes labels row and generate value to predict"""
    mat = str(column_ids['Material'][int(row.Material)]).upper()
    typ = str(column_ids['Type'][int(row.Type)])
    dirt = str(column_ids['Dirt'][int(row.Dirt)]).upper()
    # print(f'{mat=}, {typ=}, {dirt=}')

    # Cast All columns of rules to be strings
    # To handle filters that combine Booleans and *
    rules = rules.astype(str)

    # Filter by material if only one return
    mat_filter = rules[rules.Material==mat]
    if len(mat_filter) ==1: return mat_filter.Label.values[0]

    # Filter by Type
    typ_filter = mat_filter[mat_filter.Type==typ]
    if len(typ_filter) == 1: return typ_filter.Label.values[0]
    # Try all if None
    if len(typ_filter) == 0: typ_filter = mat_filter[mat_filter.Type=='*']
    if len(typ_filter) == 1: return typ_filter.Label.values[0]


    # Filter by dirt
    dirt_filter = typ_filter[typ_filter.Dirt==dirt]
    if len(dirt_filter) == 1: return dirt_filter.Label.values[0]
    if len(dirt_filter) == 0: dirt_filter = typ_filter[typ_filter.Type=='*']
    if len(dirt_filter) == 1: return dirt_filter.Label.values[0]

    # Not found in rules
    return None


def pred_column_gen(df, column_ids):
    """Generate prediction column labels based a rules table"""

    # Invert columns_ids
    inv_column_ids = {}
    for col_name, ids in column_ids.items():
        inv_column_ids[col_name] = {j:i for i,j in ids.items()}

    # Load mapping rules
    rules = pandas.read_csv(gb.RULES_PATH, sep=',')

    # Generate Ids
    uniques = list(rules.Label.unique())
    ids = dict(enumerate(uniques))
    # Invert ids
    ids_inv =  {j:i for i,j in ids.items()}
    column_ids['Y'] = ids

    # Populate the label column
    df['Y'] = None
    not_included = []
    row_remove = []
    for index, row in df.iterrows():
        label = calculate_new_label(row, column_ids, rules)
        if label:
            y = ids_inv[label] # add ordinal encoded label
            df.at[index, 'Y'] = y
        else:
            # Add not included in rules for debug
            mat = column_ids['Material'][int(row.Material)]
            typ = column_ids['Type'][int(row.Type)]
            dirt = column_ids['Dirt'][int(row.Dirt)]
            excluded = (mat, typ, dirt)
            if excluded not in not_included:
                not_included.append(excluded)
            # Add ID rows to be removed
            row_remove.append(index)

    # Remove not included labels rows
    for i in row_remove:
        df = df.drop(i)


    print(f'Labels not included: {not_included}')
    return df, column_ids

def load_filenames():
    with open(gb.FILENAMES_PATH, 'rb') as f:
        filenames = pickle.load(f)
    return filenames


def print_rules():
    rules = pandas.read_csv(gb.RULES_PATH, sep=',')
    print(rules.head())



from tqdm import tqdm
import imghdr


def split_fn(f):
    """Get ID of a image filename"""
    if f.startswith('O'): # taken with Damian script
        _id = int(f.split('.')[0].split('_')[1])
    else: # Space datasets
        _id = int(f.split('.')[0].split(' ')[0].split('-')[0])
    return _id

def get_row(df, i):
    """Get labels row by id"""
    return df.loc[df.id == i]


def get_labels_filenames(df, column_ids, split_fn=split_fn):
    """
    Get labels from image filenames.
    If filenames provided will use it otherwise will read files and return it
    """
    # Get filenames of images
    img_folder = load_filenames()

    labels = {col:[] for col in column_ids.keys()}
    labels['id'] = [] # store image ids
    err = []
    img_filenames = [] # list of images to save
    # Get labels from filename
    for filepath in tqdm(img_folder, total=len(img_folder)):
        # filepath = os.path.join(dataset_path, f)
        f = filepath.name
        if f.startswith('.'):
            continue


        # Get ID from image filename
        try:
            im_id = split_fn(f)
        except Exception as e:
            # print(f'Error getting ID of file: {f}')
            err.append(f)
            continue

        # Get labels from dataframe
        try:
            for col in labels.keys():
                label = int(get_row(df, im_id)[col])
                labels[col].append(label)

        except Exception as e:
            # print(im_id)
            err.append(f)
            continue
            # raise e

        img_filenames.append(filepath)


    print(f'Error with: {err}')
    return labels, img_filenames




def load_dataset(filepath, col='Y', train_pct=0.8, seed=1234):
    """Read the numpy features file from path"""
    with open(filepath,'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)
    
    err = []
    X = []
    Y = []
    for row in data:
        x = row[0]
        im_name = row[1]

        try:
            im_id = dataset.split_fn(im_name)
            y = int(dataset.get_row(df, im_id)[col])
        except Exception as e:
            # print(f' - {e} Error in file name: {im_name}')
            err.append(im_name)

        X.append(x)
        Y.append(y)
    
    print(f' Errors: {len(err)}')
    


    # Split train_test
    split_idx = int(train_pct * len(X))

    x_train = np.array(X[:split_idx])
    y_train = np.array(Y[:split_idx])

    x_test = np.array(X[split_idx:])
    y_test = np.array(Y[split_idx:])

    return (x_train, y_train), (x_test, y_test)




def load_image(img_path, target_size=(224, 224), preprocess_input=preprocess_input):
    """Read and preprocess image for ResNet50"""
    img = image.load_img(img_path, target_size=target_size)  # ResNet50 expects 224x224
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Preprocess for ResNet50
    return x
