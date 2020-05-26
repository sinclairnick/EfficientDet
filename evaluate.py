import argparse
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    """Evaluate accuracy of color and body classification against ground truth"""
    parser = argparse.ArgumentParser('Evaluate the performance of the vehicle detection model')
    parser.add_argument('--annotations_oath', help='Path to the ground truth annotations', required=True, type=str)
    parser.add_argument('--predictions_path', help='Path to csv file with color, body predictions', required=True, type=str)
    parser.add_argument('--classes_path', help='Path to csv file with classes', required=True, type=str)
    parser.add_argument('--colors', help='Path to csv file with colors', required=True, type=str)
    args = parser.parse_args()
    
    classes = [x[0] for x in pd.read_csv(args.classes_path, header=None).values]
    colors = [x[0] for x in pd.read_csv(args.colors_path, header=None).values]

    class_labels = {x: i for i, x in enumerate(classes)}
    color_labels = {x: i for i, x in enumerate(colors)}

    gt_data = pd.read_csv(args.annotations_path, header=None)
    gt_data.columns = ['file', 'x1', 'y1', 'x2', 'y2', 'body', 'color']
    vehicle_data = pd.read_csv(args.predictions_path)

    # assert headers are same order as classes/colors
    color_headers = [x for x in vehicle_data.columns if x.startswith('color')]
    class_headers = [x for x in vehicle_data.columns if x.startswith('body')]
    assert all([x == color_headers[i].split('/')[1] for i,x in enumerate(colors)])
    assert all([x == class_headers[i].split('/')[1] for i,x in enumerate(classes)])

    class Metric:
        def __init__(self):
            self.precision = tf.metrics.Precision()
            self.recall = tf.metrics.Recall()
            self.categorical_accuracy = tf.metrics.CategoricalAccuracy()
        def update_state(self, y_true, y_pred):
            self.precision.update_state(y_true, y_pred)
            self.recall.update_state(y_true, y_pred)
            self.categorical_accuracy.update_state(y_true, y_pred)
        def result(self):
            return pd.DataFrame({
                'precision': [self.precision.result().numpy()],
                'recall': [self.recall.result().numpy()],
                'categorical_accuracy': [self.categorical_accuracy.result().numpy()]
            })
    color_metric = Metric()
    class_metric = Metric()

    # calculate performance metrics
    for i in range(len(vehicle_data)):
        predicted = vehicle_data.iloc[i]
        ground = gt_data.iloc[i]
        ground_class_label = class_labels[ground["body"]]
        ground_color_label = color_labels[ground['color']]
        assert predicted[['file']].values[0].split('/')[-1] == ground[['file']].values[0].split('/')[-1]
        class_true = tf.one_hot(ground_class_label, depth=len(class_labels))
        color_true = tf.one_hot(ground_color_label, depth=len(color_labels))
        class_metric.update_state(class_true, predicted[class_headers])
        color_metric.update_state(color_true, predicted[color_headers])

    print('Color Results:\n', color_metric.result())
    print('Class Results:\n', class_metric.result())