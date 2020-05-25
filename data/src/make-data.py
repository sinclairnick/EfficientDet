import pandas as pd
import numpy as np
from utils import (
    get_body_names, get_colors_names,
    align_stanford_classes, remove_extraneous,
    tlhw_to_corners, save_dataset, split, mkdir,
    nzvd_pipeline, shuffle, get_colored_cars
    )
from common import input_dir, output_dir, TLHW

# stanford cars dataset modified from
# https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder#names.csv

if __name__ == '__main__':
    mkdir(output_dir)

    nzvd_train = pd.read_csv(f'{input_dir}/nzvd/train_labels.csv')
    nzvd_test = pd.read_csv(f'{input_dir}/nzvd/test_labels.csv')

    stan = pd.read_csv(f'{input_dir}/stanford-cars/anno_train.csv', header=None)
    stan.columns = ['file', *TLHW, 'class']

    nzvd_train, nzvd_test = nzvd_pipeline(nzvd_train), nzvd_pipeline(nzvd_test)

    all_data = nzvd_train.append(nzvd_test, ignore_index=True)
    all_bodies = np.unique(get_body_names(all_data))
    all_colors = np.unique(get_colors_names(all_data))

    stan = align_stanford_classes(stan, all_bodies)
    stan = shuffle(stan)

    colored_cars = get_colored_cars(all_colors)
    colored_cars = shuffle(colored_cars)

    nzvd_train, nzvd_val = split(nzvd_train)
    assert abs(len(nzvd_val) * 9 - len(nzvd_train)) < 10

    stan_train, stan_val = split(stan)
    assert abs(len(stan_val) * 9 - len(stan_train)) < 10

    colored_train, colored_val = split(colored_cars)
    assert abs(len(colored_val) * 9 - len(colored_train)) < 10

    assert np.all(np.equal(nzvd_train.columns, nzvd_test.columns)), "Column names must match"
    assert np.all(np.equal(nzvd_train.columns, stan.columns)), "Column names must match"

    save_dataset(nzvd_train, 'nzvd', 'train', 'train')
    save_dataset(nzvd_val, 'nzvd', 'train', 'val')
    save_dataset(nzvd_test, 'nzvd', 'test', 'test')
    save_dataset(stan_train, 'stanford-cars', 'train', 'train')
    save_dataset(stan_val, 'stanford-cars', 'train', 'val')
    save_dataset(colored_train, 'car-colors', 'train', 'train')
    save_dataset(colored_val, 'car-colors', 'train', 'val')

    classes_out = pd.DataFrame(np.stack([all_bodies, list(range(len(all_bodies)))], axis=1))
    colors_out = pd.DataFrame(np.stack([all_colors, list(range(len(all_colors)))], axis=1))
    classes_out.to_csv(output_dir +'/classes.csv', index=False, header=False)
    colors_out.to_csv(output_dir + '/colors.csv', index=False, header=False)
