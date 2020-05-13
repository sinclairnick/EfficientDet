import pandas as pd
import numpy as np
from utils import (
    get_body_names, get_colors_names,
    align_stanford_classes, remove_extraneous,
    tlhw_to_corners, save_dataset, split, mkdir
    )
from common import input_dir, output_dir, TLHW

if __name__ == '__main__':
    mkdir(output_dir)

    nzvd_train = pd.read_csv(f'{input_dir}/nzvd/train_labels.csv')
    nzvd_test = pd.read_csv(f'{input_dir}/nzvd/test_labels.csv')

    stan = pd.read_csv(f'{input_dir}/stanford-cars/anno_train.csv')
    stan.columns = ['file', *TLHW, 'class']

    all_data = nzvd_train.append(nzvd_test, ignore_index=True)
    all_bodies = np.unique(get_body_names(all_data))
    all_colors = np.unique(get_colors_names(all_data))

    stan = align_stanford_classes(stan, all_bodies)
    print('Stanford cars subset length: {}'.format(len(stan)))
    nzvd_train = remove_extraneous(nzvd_train)
    nzvd_test = remove_extraneous(nzvd_test)

    assert np.all(np.equal(nzvd_train.columns, nzvd_test.columns)), "Column names must match"
    assert np.all(np.equal(nzvd_train.columns, stan.columns)), "Column names must match"

    nzvd_train = tlhw_to_corners(nzvd_train)
    nzvd_test = tlhw_to_corners(nzvd_test)
    stan = tlhw_to_corners(stan)

    nzvd_train, nzvd_val = split(nzvd_train)
    assert abs(len(nzvd_val) * 9 - len(nzvd_train)) < 10

    stan_train, stan_val = split(stan)
    assert abs(len(stan_val) * 9 - len(stan_train)) < 10

    save_dataset(nzvd_train, 'nzvd', 'train', 'train')
    save_dataset(nzvd_val, 'nzvd', 'train', 'val')
    save_dataset(nzvd_test, 'nzvd', 'test', 'test')
    save_dataset(stan_train, 'stanford-cars', 'train', 'train')
    save_dataset(stan_val, 'stanford-cars', 'train', 'val')

    classes_out = pd.DataFrame(np.hstack([all_bodies, list(range(len(all_bodies)))]))
    colors_out = pd.DataFrame(np.hstack([all_colors, list(range(len(all_colors)))]))
    classes_out.to_csv(output_dir +'/classes.csv', index=False, header=False)
    colors_out.to_csv(output_dir + '/colors.csv', index=False, header=False)
