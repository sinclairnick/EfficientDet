import numpy as np
import pandas as pd
import os
import shutil
from common import input_dir, output_dir, TLHW, mappings, OUT_HEADER
from tqdm import tqdm
from cv2 import cv2

def mkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
        
def remove_extraneous(data):
    new_data = data[['file', 'top', 'left', 'height', 'width', 'body-type', 'categorical-color']]
    new_data.columns = ['file', 'top', 'left', 'height', 'width', 'body', 'color']
    return new_data

def tlhw_to_corners(data):
    top, left, height, width = [data[[col]].values for col in TLHW]
    x1, y1 = left, top
    x2, y2 = left + width, top + height
    assert np.all(x1 < x2) and np.all(y1 < y2), "All x1 must be < x2 and all y1 must be < y2"
    assert np.all(x1 >= 0) and np.all(y1 >= 0), "All x1 and y1 must be >= 0"
    data[TLHW] =  np.hstack([x1,y1,x2,y2])
    data.columns = OUT_HEADER
    return data

def get_body_names(data):
    return data[['body']].values

def get_colors_names(data):
    return data[['color']].values

def align_stanford_classes(data, all_bodies):
    data = data.values
    class_id_idx = 5
    rows = pd.read_csv(f'{input_dir}/stanford-cars/names.csv', header=None).values
    bodies = pd.Series([row[0].split(' ')[-2].lower() for row in rows])
    
    # coerce some names
    for old, new in mappings.items():
        bodies = bodies.replace(old, new)
    
    bodies = bodies.values
    # replace class id with class name
    class_ids = data[:,class_id_idx].astype(int)
    class_names = bodies[class_ids -1]
    data[:,class_id_idx] = class_names

    filtered_data = [data[i] for i in range(len(data)) if data[i,class_id_idx] in all_bodies]
    
    dummy_color =  np.repeat('black', (len(filtered_data)))
    filtered_data = np.hstack([filtered_data, np.expand_dims(dummy_color,1)])
    assert np.all([x in all_bodies for x in filtered_data[:,class_id_idx]]), "Bodies must all be in the specified set of bodies"

    # set bounding boxes correctly
    filtered_data = pd.DataFrame(filtered_data)
    filtered_data.columns = OUT_HEADER
    return filtered_data


def split(data, prop=0.9):
    split_point = int(len(data) * prop)
    return data.iloc[:split_point], data.iloc[split_point:]

def shuffle(data):
    return data.sample(frac=1).reset_index(drop=True) # "drop" prevents old index from being prepended to columns

def get_colored_cars(all_colors):
    fnames = np.expand_dims(os.listdir(f'{input_dir}/car-colors/train'),1 )
    # set dummy bbox such that [x1,y1] < [x2,y2]
    dummy_bbox = np.expand_dims([20,20,200,200], 0)
    dummy_bboxes = np.repeat(dummy_bbox, (len(fnames)), 0)
    dummy_body = np.expand_dims(np.repeat('coupe', (len(fnames))),1)
    colors = np.expand_dims(list(map(lambda x: x[0].split('_')[0], fnames)), 1)
    assert np.all([x in all_colors for x in colors]), "All colors must be one of {}".format(all_colors)
    data = pd.DataFrame(np.hstack([fnames, dummy_bboxes, dummy_body, colors]))
    data.columns = OUT_HEADER
    return data


def copy_images(data, in_dir, out_dir):
    data.reset_index(drop=True, inplace=True)
    for idx, row in enumerate(tqdm(data.values)):
        fname = row[0]
        x1, y1, x2, y2 = [int(x) for x in row[1:5]]
        Xs = np.array([x1,x2])
        Ys = np.array([y1,y2])
        
        in_path = f'{in_dir}/{fname}'
        out_path = f'{out_dir}/{fname}'

        img = cv2.imread(in_path)
        H, W = img.shape[:2]

        if (
            not np.all(np.hstack([Xs, Ys]) >= 0) or
            not np.all(Xs <= W) or 
            not np.all(Ys <= H)
        ):
            print('Dropped row with out of bounds bbox: {}'.format(fname))
            data.drop([idx], inplace=True)
        cv2.imwrite(out_path, img)
    return data
        
def save_dataset(data, ds_name, folder, set_name):
    out_dir = output_dir + '/' + ds_name
    img_dir = out_dir + '/' + folder

    for dir_ in [out_dir, img_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # copy images to outfolder
    data = copy_images(data, f'{input_dir}/{ds_name}/{folder}', img_dir)
    
    # append folder/ to fname
    data[['file']] = folder + '/' + data[['file']]

    # save csv
    data.to_csv(f'{out_dir}/{set_name}_annotations.csv', index=False, header=False)

def merge_4x4s(data):
    data = data.replace('4x4', 'suv')
    return data

def merge_greys(data):
    data = data.replace('grey', 'silver')
    return data

def nzvd_pipeline(data):
    data = remove_extraneous(data)
    data = tlhw_to_corners(data)
    data = merge_4x4s(data)
    data = merge_greys(data)
    data = shuffle(data)
    return data