import argparse
import pandas as pd
import tensorflow as tf
from tqdm import trange
import json
import time
import os

OUT_DIR = 'evaluations'

class Metric:
    def __init__(self, name):
        self.name = name
        self.precision = tf.metrics.Precision()
        self.recall = tf.metrics.Recall()
        self.categorical_accuracy = tf.metrics.CategoricalAccuracy()
    def update_state(self, y_true, y_pred):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)
        self.categorical_accuracy.update_state(y_true, y_pred)
    def result(self):
        return dict(
            name=self.name,
            precision=self.precision.result().numpy().item(),
            recall=self.recall.result().numpy().item(),
            categorical_accuracy=self.categorical_accuracy.result().numpy().item()
        )
        

if __name__ == '__main__':
    """Evaluate accuracy of color and body classification against ground truth"""
    parser = argparse.ArgumentParser('Evaluate the performance of the vehicle detection model')
    parser.add_argument('--annotations_path', help='Path to the ground truth annotations', required=True, type=str)
    parser.add_argument('--predictions_path', help='Path to csv file with color, body predictions', required=True, type=str)
    parser.add_argument('--classes_path', help='Path to csv file with classes', required=True, type=str)
    parser.add_argument('--colors_path', help='Path to csv file with colors', required=True, type=str)
    args = parser.parse_args()
    
    classes = [x[0] for x in pd.read_csv(args.classes_path, header=None).values]
    colors = [x[0] for x in pd.read_csv(args.colors_path, header=None).values]

    class_labels = {x: i for i, x in enumerate(classes)}
    color_labels = {x: i for i, x in enumerate(colors)}

    gt_data = pd.read_csv(args.annotations_path, header=None)
    if 'train' in args.annotations_path.lower(): # get validation annotations too if train set
        val_data = pd.read_csv(args.annotations_path.replace('train', 'val'), header=None)
        gt_data = pd.concat([gt_data, val_data])
    gt_data.columns = ['file', 'x1', 'y1', 'x2', 'y2', 'body', 'color']
    gt_data = gt_data.sort_values(by=['file'])
    gt_data = gt_data.reset_index()
    vehicle_data = pd.read_csv(args.predictions_path)
    vehicle_data = vehicle_data.sort_values(by=['file'])

    # assert headers are same order as classes/colors
    color_headers = [x for x in vehicle_data.columns if x.startswith('color')]
    class_headers = [x for x in vehicle_data.columns if x.startswith('body')]
    assert all([x == color_headers[i].split('/')[1] for i,x in enumerate(colors)])
    assert all([x == class_headers[i].split('/')[1] for i,x in enumerate(classes)])

    color_metric = Metric('color')
    class_metric = Metric('body')

    # calculate performance metrics
    for i in trange(len(vehicle_data)):
        predicted = vehicle_data.iloc[i]
        ground = gt_data.iloc[i]
        ground_class_label = class_labels[ground["body"]]
        ground_color_label = color_labels[ground['color']]

        pred_fname = predicted[['file']].values[0].split('/')[-1]
        gt_fname = ground[['file']].values[0].split('/')[-1]
        assert pred_fname == gt_fname

        class_true = tf.one_hot(ground_class_label, depth=len(class_labels))
        color_true = tf.one_hot(ground_color_label, depth=len(color_labels))
        class_metric.update_state(class_true, predicted[class_headers])
        color_metric.update_state(color_true, predicted[color_headers])

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open('{}/evaluation_{}.json'.format(OUT_DIR, time.time()), 'w+') as f:
        out = dict(
            predictions_path=args.predictions_path,
            annotations_path=args.annotations_path,
            color_results=color_metric.result(),
            body_results=class_metric.result(),
            # to be filled out manually
            notes=[],
            phi='',
            train_data=[]
        )
        print(out)
        f.write(json.dumps(out))