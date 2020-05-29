import cv2
import json
import numpy as np
import os
import time
import glob
import pandas as pd

import argparse
from tensorflow import keras

from model import efficientLPR
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

import json
from tqdm import tqdm

WEIGHTED_BIFPN = True
IMAGE_SIZES = (512, 640, 768, 896, 1024, 1280, 1408)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description="""
    Script to infer on a folder of images.
    Weights paths are assumed to be in weights/
    """)
    parser.add_argument('--phi', help="Phi model number", default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--class_path', help="Csv path to detection classes", type=str, required=True)
    parser.add_argument('--colors_path', help="Csv path to vehicle colors", type=str, required=True)
    parser.add_argument('--score_thresh', help="Score threshold for detections", default=0.1, type=float)
    parser.add_argument('--model_path', help="Path to model weights", default='weights/extracted-weights-phi0.h5', type=str)
    parser.add_argument('--image_dir', help="Path to input image directory", required=True, type=str)
    args = parser.parse_args()

    phi = args.phi
    score_threshold = args.score_thresh
    image_size = IMAGE_SIZES[phi]
    model_path = args.model_path

    classes = [x[0] for x in pd.read_csv(args.class_path, header=None).values]
    color_classes = [x[0] for x in pd.read_csv(args.colors_path, header=None).values]
    num_colors = len(color_classes)
    num_classes = len(classes)
    print('Num classes', num_classes)
    print('Num colors', num_colors)
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientLPR(phi=phi,
                            weighted_bifpn=WEIGHTED_BIFPN,
                            num_classes=num_classes,
                            num_colors=num_colors,
                            score_threshold=score_threshold)
        
    model.load_weights(model_path, by_name=True)

    predictions = []

    image_files = sorted(glob.glob(f'{args.image_dir}/*.jp*'))
    for image_path in tqdm(image_files):
        image = cv2.imread(image_path)
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        out = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = out[0]
        color_preds = out[1].numpy()
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        boxes = postprocess_boxes(boxes=boxes.copy(), scale=scale, height=h, width=w)

        # get best bbox and all class predictions for csv
        i_best = np.argmax(scores)
        best_bbox = boxes[i_best]
        best_label = labels[i_best]
        best_color = color_classes[np.argmax(color_preds)]
        
        # # select indices which have a score above the threshold
        # indices = np.where(scores[:] > score_threshold)[0]

        # # select those detections
        # boxes = boxes[indices]
        # labels = labels[indices]

        # draw_boxes(src_image, boxes, scores, labels, colors, classes)
        draw_boxes(src_image, [best_bbox], [scores[i_best]], [best_label], colors, classes)
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topRightCorner         = (10,50)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2

        cv2.putText(src_image, best_color, 
            topRightCorner,  
            font, 
            fontScale,
            fontColor,
            lineType)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', src_image)
        # cv2.imwrite(f'tmp/{image_path.split("/")[-1]}', src_image)
        # cv2.waitKey(0)

        # predictions.append([image_path, *[int(x) for x in best_bbox], classes[best_label], best_color])
        body_preds = np.zeros((num_classes))

        scores = np.array(scores)
        labels = np.array(labels)
        positive_idxs = np.where(scores > 0)
        # only save positive scores above score threshold
        # reversed so greater scores overwrite lower scores of same class
        for label, score in reversed(list(zip(labels[positive_idxs], scores[positive_idxs]))):
            body_preds[label] = score

        # print(best_bbox, body_preds, color_preds)
        prediction = [image_path, *best_bbox, *body_preds, *color_preds[0]]
        predictions.append(prediction)

    out_df = pd.DataFrame(predictions)
    out_df.columns = ['file', 'x1', 'y1', 'x2', 'y2', *[f'body/{x}' for x in classes], *[f'color/{x}' for x in color_classes]]

    out_df.to_csv('predictions_phi{}_{}.csv'.format(phi, args.image_dir.split('/')[-2]), index=False)


if __name__ == '__main__':
    main()