import numpy as np
import pandas as pd
import os
import shutil
from common import input_dir, output_dir, TLHW, mappings

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
    data.columns = ['file', 'x1', 'y1', 'x2', 'y2', 'body', 'color']
    return data

def get_body_names(data):
    return data[['body-type']].values

def get_colors_names(data):
    return data[['categorical-color']].values

def split(data, prop=0.9):
    split_point = int(len(data) * prop)
    return data.iloc[:split_point], data.iloc[split_point:]

def copy_images(data, in_dir, out_dir):
    for fname in data[['file']].values.squeeze():
        shutil.copy2(f'{in_dir}/{fname}', f'{out_dir}/{fname}')

def save_dataset(data, ds_name, folder, set_name):
    out_dir = output_dir + '/' + ds_name
    img_dir = out_dir + '/' + folder

    for dir_ in [out_dir, img_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # copy images to outfolder
    copy_images(data, f'{input_dir}/{ds_name}/{folder}', img_dir)
    

    # append folder/ to fname
    data[['file']] = folder + '/' + data[['file']]

    # save csv
    data.to_csv(f'{out_dir}/{set_name}_annotations.csv', index=False, header=False)

def mkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)

def align_stanford_classes(data, all_bodies):
    data = data.values
    class_id_idx = 5
    rows = pd.read_csv(f'{input_dir}/stanford-cars/names.csv', header=None).values
    bodies = np.array([row[0].split(' ')[-2].lower() for row in rows])
    
    # coerce some names
    for old, new in mappings.items():
        idxs = np.where(bodies == old)
        bodies[idxs] = new
    
    # replace class id with class name
    data[:,class_id_idx] = bodies[data[:,class_id_idx].astype(int) -1]
    filtered_data = [data[i] for i in range(len(data)) if data[i,class_id_idx] in all_bodies]
    
    dummy_color =  np.repeat('black', (len(filtered_data)))
    filtered_data = np.hstack([filtered_data, np.expand_dims(dummy_color,1)])
    assert np.all([x in all_bodies for x in filtered_data[:,class_id_idx]]), "Bodies must all be in the specified set of bodies"
    filtered_data = pd.DataFrame(filtered_data, columns=['file', *TLHW, 'body', 'color'])
    return filtered_data