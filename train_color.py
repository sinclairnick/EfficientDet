from model import efficientLPR
import pandas as pd
import argparse
from losses import categorical_focal_loss
import wandb
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES
from datetime import date
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizer import Adam
from color_classifier import create_color_classifier
from efficientnet import (
    EfficientNetB0, EfficientNetB1,EfficientNetB2, EfficientNetB3,
      EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7 
      )

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def create_generators():
    # TODO;
    return train_generator, val_generator

def main(args=None):
    """
    Creates a model with only efficient net and color classifier. Used to get the color weights IN ISOLATION ONLY.
    Largely a copy of the normal training script.
    Allows us to use data without bboxes and to use tf.dataset multiprocessing.
    """
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations_path', help='Path to CSV file containing annotations for training.')

    csv_parser.add_argument('colors_path', help='Path to a CSV file containing colors label mapping.')

    parser.add_argument('--dropout_rate', help='Dropout rate for classification branch', default=0.1, type=float, choices=(0.1, 0.2, 0.3, 0.4, 0.5))
    parser.add_argument('--wandb', help='Whether to use wandb syncing', default=False, action="store_true")
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)

    csv_parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')

    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--freeze-bn', help='Freeze training of BatchNormalization layers.', action='store_true')
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true')

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')



    args = parser.parse_args()
    phi = args.phi
    freeze_bn = args.freeze_bn

    dropout = 0.5

    num_classes = pd.read_csv(args.classes_path, header=None).values.shape[0]
    num_colors = pd.read_csv(args.colors_path, header=None).values.shape[0]

    if args.wandb:
        wandb.init(config=args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    image_input = keras.layers.Input(input_shape)
    backbone_cls = backbones[phi]

    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)

    feature_inputs = [keras.layers.Input(tf.squeeze(feature, axis=0).shape) for feature in features]

    color_classifier = create_color_classifier(feature_inputs, num_colors, dropout)
    colors_out = color_classifier(features)

    model = keras.models.MOdel(inputs=[image_input], outputs=[colors_out])


    # load pretrained weights    
    if args.snapshot:
        if args.snapshot == 'imagenet':
            print('Loading imagenet weights')
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            model.load_weights(args.snapshot, by_name=True)

    backbone_layers = [model.layers[i] for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi])]

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for layer in backbone_layers:
            layer.trainable = False

    if args.gpu and len(args.gpu.split(',')) > 1:
        model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    # compile model
    model.compile(optimizer=Adam(lr=args.lr), loss=[categorical_focal_loss()], metrics=['categorical_accuracy'])

    return model.fit(
        train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=[wandb.keras.WandbCallback()],
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        validation_data=validation_generator
    )

    if args.wandb:
        wandb.save('checkpoints/**/*.h5')


if __name__ == '__main__':
    main()
