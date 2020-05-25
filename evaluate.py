import argparse
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate the performance of the vehicle detection model')
    parser.add_argument('--annotations_csv', help='Path to the ground truth annotations', required=True, type=str)
    parser.add_argument('--car_csv', help='Path to csv file with color, body predictions', required=True, type=str)
    parser.add_argument('--lp_csv', help='path to csv with lp character predictions', required=True, type=str)
    args = parser.parse_args()

    gt_data = pd.read_csv(args.annotations_csv, header=None).values
    vehicle_data = pd.read_csv(args.car_csv, header=None).values
    lp_data = pd.read_csv(args.lp_csv, header=None).values

    # use .update_state() with these metrics
    precision = tf.metrics.Precision()
    recall = tf.metrics.Recall()
    categorical_accuracy = tf.metrics.CategoricalAccuracy()
    
    # 0 wrong, 1 wrong, ...
    lp_accuracies = [tf.metrics.Accuracy() for x in range(7)]

    # TODO: MERGE VEHICLE AND LP DATA
    merged_data = []

    # calculate performance metrics
    for row in zip(merged_data, gt_data):
        # TODO: compare the merged data to the ground truth
        # COLOR/BODY: calculate precision, accuracy, whatever
        # LP: calculate accuracy at different levels, e.g. 1 wrong, 2 wrong etc. to show how many are close but off
        pass

    # import similarity metric
    # evaluate matching/unmatching classifications to GT, wrt. similarty metric