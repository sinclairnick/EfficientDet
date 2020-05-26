from tensorflow import keras
import argparse
from model import efficientLPR
import pandas as pd

import tensorflow as tf

def freeze_model(model):
    # freeze whole model
    for layer in model.layers:
        layer.trainable = False

if __name__ == '__main__':
    """
    Script used to merge the trained portions of seperately trained
    car detection and color classifier weights.
    Also works around a Keras bug involving frozen weights loading weirdly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_weights', help='full model weights with color trained', type=str, required=True)
    parser.add_argument('--body_weights', help='full model weights with body trained', type=str, required=True)
    parser.add_argument('--phi', help="Phi model number", default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--class_path', help="Csv path to detection classes", type=str, required=True)
    parser.add_argument('--colors_path', help="Csv path to vehicle colors", type=str, required=True)
    parser.add_argument('--score_thresh', help="Score threshold for detections", default=0.2, type=float)
    args = parser.parse_args()

    phi = args.phi
    score_threshold = args.score_thresh

    classes = [x[0] for x in pd.read_csv(args.class_path, header=None).values]
    color_classes = [x[0] for x in pd.read_csv(args.colors_path, header=None).values]
    num_colors = len(color_classes)
    num_classes = len(classes)
    
    _, model = efficientLPR(phi=phi,
                        weighted_bifpn=True,
                        num_classes=num_classes,
                        num_colors=num_colors,
                        score_threshold=score_threshold)

    # unfreeze body and load its weights
    freeze_model(model)
    model.get_layer('car_detector').trainable = True
    model.load_weights('weights/' + args.body_weights)
    body_weights = model.get_layer('car_detector').get_weights()

    # unfreeze color and load its weights
    freeze_model(model)
    model.get_layer('color_classifier').trainable = True
    model.load_weights('weights/' + args.color_weights)
    color_weights = model.get_layer('color_classifier').get_weights()

    # set color/body weights
    model.get_layer('car_detector').set_weights(body_weights)
    model.get_layer('color_classifier').set_weights(color_weights)

    # unfreeze model
    for layer in model.layers:
        layer.trainable = True

    model.save_weights('weights/extracted-weights.h5')
    # tf.saved_model.save(model, 'saved_model') # save to tf format

