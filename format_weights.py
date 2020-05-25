from tensorflow import keras
import argparse
from model import efficientLPR
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to full model trained weights', type=str, required=True)
    parser.add_argument('--feature', help='whether to extract the color or body weights', choices=('color', 'body'), required=True)
    parser.add_argument('--phi', help="Phi model number", default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--class_path', help="Csv path to detection classes", type=str, required=True)
    parser.add_argument('--colors_path', help="Csv path to vehicle colors", type=str, required=True)
    parser.add_argument('--score_thresh', help="Score threshold for detections", default=0.2, type=float)
    args = parser.parse_args()

    identifier = 'car_detector' if args.feature == 'body' else 'color_classifier'

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

    # freeze whole model
    for layer in model.layers:
        layer.trainable = False

    model.get_layer(identifier).trainable = True

    model.load_weights(args.model_path)

    # unfreeze model
    for layer in model.layers:
        layer.trainable = True

    model.save_weights('extracted-weights.h5')

