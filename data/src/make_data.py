import os
from scipy.io import loadmat
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm

def split(dataset):
    # train/test
    split = int(dataset.shape[0] * 0.8)
    indices = np.random.permutation(dataset.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]
    train_annos, test_annos = dataset[train_idx], dataset[test_idx]
    # train/val
    split = int(train_annos.shape[0] * 0.9)
    indices = np.random.permutation(train_annos.shape[0])
    train_idx, val_idx = indices[:split], indices[split:]
    train_annos, val_annos = train_annos[train_idx], train_annos[val_idx]
    return train_annos, val_annos, test_annos

def save_csvs(train, val, test, classes, out_dir):
    pd.DataFrame(train).to_csv(out_dir + '/train_annotations.csv', header=None, index=None)
    pd.DataFrame(val).to_csv(out_dir + '/val_annotations.csv', header=None, index=None)
    pd.DataFrame(test).to_csv(out_dir + '/test_annotations.csv', header=None, index=None)

    classes = np.hstack([np.expand_dims(classes,1), np.expand_dims(range(len(classes)), 1)])
    pd.DataFrame(classes).to_csv(out_dir + '/colors.csv', header=None, index=None)



def main():
    parser = argparse.ArgumentParser('Makes data from CompCars dataset for training')
    parser.add_argument('--in_dir', help="input data directory", default='data/raw/comp-cars', type=str)
    args = parser.parse_args()

    in_dir = args.in_dir
    np.random.seed(1)

    # ------------------------- SURVEILENCE (COLOR) DATA ------------------------- #
    in_dir = 'data/raw/comp-cars'
    surv_dir = in_dir + '/sv_data'
    label_mat = loadmat(surv_dir + '/color_list.mat')["color_list"]

    colors = ["black", "white", "red", "yellow", "blue", "green", "purple", "brown", "champagne", "silver"]

    # color_labels: [n, [fname, color]]
    color_annos = [['image/' + x[0][0], colors[x[1][0][0]]] for x in label_mat if not x[1][0][0] == -1 ] # omit unrecognized colors
    color_annos = np.array(color_annos)

    save_csvs(*split(color_annos), classes=colors, out_dir=surv_dir)

    # --------------------------------- WEB DATA --------------------------------- #
    web_dir = in_dir + '/data'
    label_dir = web_dir + '/label'
    side_view_id = 3

    makes = [x[0][0] for x in loadmat(web_dir + '/misc/make_model_name.mat')["make_names"]]
    makes[makes.index('Lamorghini ')] = 'Lamborghini' # fix type in DS

    def get_annos(annos):
        for make_id in tqdm(sorted(os.listdir(label_dir))):
            make_dir = '{}/{}'.format(label_dir, make_id)

            for model_id in sorted(os.listdir(make_dir)):
                model_path = '{}/{}'.format(make_dir, model_id)

                for year in sorted(os.listdir(model_path)):
                    leaf_path = '{}/{}'.format(model_path, year)
                    
                    for fname in sorted(os.listdir(leaf_path)):
                        file_path = '{}/{}'.format(leaf_path, fname)
                        with open(file_path, 'r') as f:
                            file_data = f.read().strip().split('\n')
                            viewpoint_id = int(file_data[0])
                            bbox = file_data[2].split(' ')
                            if not (viewpoint_id == side_view_id):
                                annos.append(["image/" + "/".join(file_path[:-4].split("/")[5:]) + '.jpg', bbox[0], bbox[1], bbox[2], bbox[3], makes[int(make_id)-1]])
        return annos
    # annos: [n, [fname (of image), make, year, x1, y1, x2, y2]]
    annos = np.array(get_annos([]))
    annos = np.hstack([annos, np.repeat('black', annos.shape[0]).reshape(-1,1)]) # add dummy color column

    save_csvs(*split(annos), classes=makes, out_dir=web_dir)

if __name__ == '__main__':
    main()