import pandas as pd
import numpy as np
import re

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/sample_submission.csv')

    return train, test

def split(df):
    image_ids = df['image_id'].unique()
    val_ids = image_ids[-665:]
    train_ids = image_ids[:-665]
    val = df[df['image_id'].isin(val_ids)]
    train = df[df['image_id'].isin(train_ids)]

    return train, val

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def create_data(data_folder):
    train, test = read_data(data_folder)

    train['x'] = -1
    train['y'] = -1
    train['w'] = -1
    train['h'] = -1

    train[['x', 'y', 'w', 'h']] = np.stack(train['bbox'].apply(lambda x : expand_bbox(x)))
    train.drop(columns = ['bbox'], inplace = True)
    train['x'] = train['x'].astype(np.float)
    train['y'] = train['y'].astype(np.float)
    train['w'] = train['w'].astype(np.float)
    train['h'] = train['h'].astype(np.float)

    train, val = split(train)

    return train, val, test