"""
Script that transforms the predictions into three features:
    lp_distance: a similarity metric representing the levenshtein distance between true and predicted LP
    body_distance: distance between true and predicted body
    color_distance: distance between true and predicted color

"""

import os
os.chdir('/Users/nick/Documents/school/research/EfficientLPR')
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from rapidfuzz import fuzz
import pickle
from sklearn.linear_model import LinearRegression
import argparse

betas = [10, -1, -1]


def get_l2_norm(y_trues, y_preds):
    distances = []
    for i in range(len(y_preds)): # calculate euclidian distance between preds and answer
        y_true, y_pred = y_trues[i], y_preds[i]
        distances.append(np.linalg.norm(y_true - y_pred))
    return np.expand_dims(distances,1)

def get_lev_distance(y_true, y_preds):
    lev_distances = []
    for row in np.hstack([np.expand_dims(y_preds, 1), np.expand_dims(y_true, 1)]):
        lev_distances.append(fuzz.ratio(row[0], row[1]))
    return np.expand_dims(lev_distances,1)

def similarity(x):
    return np.dot(x, betas)


def main():
    parser = argparse.ArgumentParser('Evaluate model using similarity matching')
    parser.add_argument('--test', help='whether to use test or train data', default=False, action="store_true")
    parser.add_argument('--threshold', help='similarity threshold', default=0.4, type=int)
    args = parser.parse_args()

    test = args.test
    threshold = args.threshold


    predictions_path = f'predictions_nzvd_{"test" if test else "train"}.full.csv'
    trues_path = f'data/processed/nzvd/{"test" if test else "train"}_annotations.csv'
    lps_path = f'data/raw/nzvd/{"test" if test else "train"}_labels.csv'
    classes_path = 'data/processed/classes.csv'
    colors_path = 'data/processed/colors.csv'

    # get class data
    classes = [x[0] for x in pd.read_csv(classes_path, header=None).values]
    colors = [x[0] for x in pd.read_csv(colors_path, header=None).values]
    class_labels = {x:i for i,x in enumerate(classes)}
    color_labels = {x:i for i,x in enumerate(colors)}

    # get y_pred 
    preds = pd.read_csv(predictions_path)
    preds = preds.fillna('') # fill NaN values with empty string

    # get y_true
    lps = pd.read_csv(lps_path)[['lp-string']].T.squeeze()
    lps = lps.apply(lambda x: str(x).replace(' ', ''))
    trues = pd.read_csv(trues_path, header=None)
    if 'train' in trues_path:
        trues = pd.concat([trues, pd.read_csv(trues_path.replace('train', 'val'), header=None)])
    trues.columns = ['file', 't', 'l', 'h', 'w', 'body', 'color']
    trues = trues.sort_values(by=['file'])
    trues.reset_index(inplace=True)
    trues = trues.assign(lp=lps)


    # LICENSE PLATES    
    lp_true, lp_pred = trues[['lp']].values.squeeze(), preds[['lp']].values.squeeze()
    lp_acc = np.mean([lp_true == lp_pred])
    print("LP Exact Accuracy:", f'{lp_acc}%', )

    def featurize(trues, preds):
        """Converts [preds, true] into [levenshtein distance, CCE_body, CCE_color]"""

        # levenshtein distance of license plates
        lev_distances = get_lev_distance(trues[['lp']].values.squeeze(), preds[['lp']].values.squeeze())

        # BODY
        body_true = list(map(lambda x: class_labels[x], trues[['body']].values.squeeze().tolist()))
        body_true = tf.one_hot(body_true, depth=len(class_labels))
        body_headers = [header for header in preds.columns if header.startswith('body')]
        body_pred = preds[body_headers].values
        # body_pred = (np.argmax(body_pred, axis=1) == np.expand_dims(body_true, 0)).T
        body_cce = np.expand_dims(tf.losses.categorical_crossentropy(body_true, body_pred).numpy(), 1)


        # COLOR
        color_true = list(map(lambda x: color_labels[x], trues[['color']].values.squeeze().tolist()))
        color_true = tf.one_hot(color_true, depth=len(color_labels))
        color_headers = [header for header in preds.columns if header.startswith('color')]
        color_pred = preds[color_headers].values
        # color_pred = (np.argmax(color_pred, axis=1) == np.expand_dims(color_true,0)).T
        color_cce = np.expand_dims(tf.losses.categorical_crossentropy(color_true, color_pred).numpy(), 1)

        return [lev_distances/100, color_cce, body_cce]

    y_positive = np.expand_dims(np.repeat([1], len(preds)), 1) # positive samples have class==1
    x_positive = np.hstack(featurize(trues, preds))
    x_negative = np.empty((0,3))
    y_negative = np.empty((0,1))

    # create negatives
    for i, sample in enumerate(trues.iloc):
        """ Create negative samples. For each true sample, pair with every non-matching sample """
        headers = sample.index.values
        neg_true = [sample.values for _ in range(len(trues)-1)]
        neg_true = pd.DataFrame(neg_true, columns=headers)
        # add all preds except current sample
        neg_pred = pd.concat([preds.iloc[:i], preds.iloc[i+1:]])
        neg_x = featurize(neg_true, neg_pred)
        neg_x = np.hstack(neg_x)
        neg_y = np.zeros((len(neg_x), 1))

        x_negative = np.concatenate([x_negative, neg_x])
        y_negative = np.concatenate([y_negative, neg_y])

    x_negative = np.array(x_negative)
    y_negative = np.array(y_negative)
    # balance data
    neg_idxs = np.random.randint(0, len(x_positive), len(x_positive))
    x_negative = x_negative[neg_idxs]
    y_negative = y_negative[neg_idxs]

    x = np.vstack([x_positive, x_negative])
    y = np.vstack([y_positive, y_negative])

    score_accs = []

    # compare each prediction against entire GT "database"
    for i, prediction in enumerate(preds.iloc):
        # initialize y_array
        y_true = np.zeros(len(trues))
        # set single true match
        y_true[i] = 1
        # repeat sample
        colnames = prediction.index.values
        prediction = prediction.values
        x_pred = [prediction for _ in range(len(trues))]
        x_pred = pd.DataFrame(x_pred, columns=colnames)

        # compare sample against all samples
        x_pred = featurize(trues, x_pred)
        x_pred = np.hstack(x_pred)

        score = similarity(x_pred)
        imax = np.argmax(score)
        score_accs.append(y_true[imax] == 1)
    print('Score acc', np.mean(score_accs))

    # evaluate on 50% positive and 50% negative samples
    prec = tf.metrics.Precision()
    rec = tf.metrics.Recall()
    acc = tf.metrics.Accuracy()

    for x_, y_ in zip(x, y):
        if similarity(x_) > threshold:
            prec.update_state(y_,[1])
            rec.update_state(y_,[1])
            acc.update_state(y_,[1])
        else:
            prec.update_state(y_,[0])
            rec.update_state(y_,[0])
            acc.update_state(y_,[0])
    print('Precision', prec.result().numpy())
    print('Recall', rec.result().numpy())
    print('Accuracy', acc.result().numpy())