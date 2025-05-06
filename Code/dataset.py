"""
Dataset related function

JCA
"""
import pickle

import pandas
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pandas.set_option('display.width', 200)  # or some large value
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_colwidth', None)


NAN_TAG = 'Unknown'


def load_labels(labels_path, materials_replace=None):
    """Load CSV with SASVAR labels"""
    # Load labels
    nan_tag = 'Unknown'

    # Target columns
    column_ids = {'Material':{}, 'Brand':{}, 'Type':{},
                  'Dirt':{}, 'Package Color':{}, 'Bottle Cap':{},
                  'ProductType':{}}

    df = pandas.read_csv(labels_path, sep=',')
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
        ids[-1] = NAN_TAG

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


def pred_column_gen(df, column_ids, rules_path):
    """Generate prediction column labels based a rules table"""

    # Invert columns_ids
    inv_column_ids = {}
    for col_name, ids in column_ids.items():
        inv_column_ids[col_name] = {j:i for i,j in ids.items()}

    # Load mapping rules
    rules = pandas.read_csv(rules_path, sep=',')

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

def load_filenames(repo_path):
    filenames_path = f'{repo_path}/app_image_filenames.pickle'
    with open(filenames_path, 'rb') as f:
        filenames = pickle.load(f)
    return filenames