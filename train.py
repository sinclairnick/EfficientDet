"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date
import os
import sys
import tensorflow as tf

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientLPR
from losses import smooth_l1, focal, smooth_l1_quad, categorical_focal_loss
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

# NOTE: ADDED
import wandb


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    # if args.tensorboard_dir:
    #     if tf.version.VERSION > '2.0.0':
    #         file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
    #         file_writer.set_as_default()
    #     tensorboard_callback = keras.callbacks.TensorBoard(
    #         log_dir=args.tensorboard_dir,
    #         histogram_freq=0,
    #         # batch_size=args.batch_size, # this argument is deprecated in tf2.0
    #         write_graph=True,
    #         write_grads=False,
    #         write_images=False,
    #         embeddings_freq=0,
    #         embeddings_layer_names=None,
    #         embeddings_metadata=None
    #     )
    #     callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from eval.coco import Evaluate
            # use prediction model for evaluation
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        else:
            from eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if args.compute_val_loss
                else f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'
            ),
            verbose=1,
            save_weights_only=True,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)
    if args.wandb:
        callbacks.append(wandb.keras.WandbCallback())
    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
        'detect_text': args.detect_text,
        'detect_quadrangle': args.detect_quadrangle
    }

    # create random transform generator for augmenting training data
    print('Random augmentation: {}'.format('enabled' if args.random_transform else 'disabled'))
    if args.random_transform:
        # reduce intensity of color augmentation when color branch is training
        color_factor = 0.9 if not args.freeze_color else None

        misc_effect = MiscEffect()
        visual_effect = VisualEffect(color_factor=color_factor)
    else:
        misc_effect = None
        visual_effect = None

    if args.dataset_type == 'pascal':
        from generators.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            skip_difficult=True,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'val',
            skip_difficult=True,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        from generators.csv_ import CSVGenerator
        train_generator = CSVGenerator(
            args.annotations_path,
            args.classes_path,
            args.colors_path,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        if args.val_annotations_path:
            validation_generator = CSVGenerator(
                args.val_annotations_path,
                args.classes_path,
                args.colors_path,
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from generators.coco import CocoGenerator
        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            group_method='random',
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.gpu and parsed_args.batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))

    return parsed_args


def parse_args(args):
    """
    Parse the arguments.
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
    csv_parser.add_argument('classes_path', help='Path to a CSV file containing class label mapping.')

    # NOTE: ADDED
    csv_parser.add_argument('colors_path', help='Path to a CSV file containing colors label mapping.')

    # NOTE: ADDED
    parser.add_argument('--dropout_rate', help='Dropout rate for classification branch', default=0.1, type=float, choices=(0.1, 0.2, 0.3, 0.4, 0.5))
    parser.add_argument('--hinge_loss', help='Whether to use hinge loss as activation', default=False, action="store_true")
    parser.add_argument('--wandb', help='Whether to use wandb syncing', default=False, action="store_true")
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('--freeze_color', help='Whether to freeze color classification', default=False, action="store_true")
    parser.add_argument('--freeze_body', help='Whether to freeze car detection', default=False, action="store_true")

    csv_parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--detect-quadrangle', help='If to detect quadrangle.', action='store_true', default=False)
    parser.add_argument('--detect-text', help='If is text detection task.', action='store_true', default=False)

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
    # parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
    #                     default='logs/{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    train_generator, validation_generator = create_generators(args)

    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors
    num_colors = train_generator.num_colors() # NOTE: ADDED

    # NOTE: ADDED
    if args.wandb:
        wandb.init(config=args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    model, prediction_model = efficientLPR(args.phi,
                                        num_classes=num_classes,
                                        num_anchors=num_anchors,
                                        num_colors=num_colors,
                                        dropout_rate=args.dropout_rate,
                                        hinge_loss=args.hinge_loss,
                                        weighted_bifpn=args.weighted_bifpn,
                                        freeze_bn=args.freeze_bn,
                                        detect_quadrangle=args.detect_quadrangle
                                        )
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

    dummy_loss = lambda x, y: float(0)
    classification_loss = focal()
    regression_loss = smooth_l1_quad() if args.detect_quadrangle else smooth_l1()
    colors_loss = categorical_focal_loss()

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for layer in backbone_layers:
            layer.trainable = False
    
    if args.freeze_body:
        classification_loss, regression_loss = dummy_loss, dummy_loss
        model.get_layer('car_detector').trainable = False

    if args.freeze_color:
        colors_loss = dummy_loss
        model.get_layer('color_classifier').trainable = False

    if args.gpu and len(args.gpu.split(',')) > 1:
        model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    # compile model
    model.compile(optimizer=Adam(lr=args.lr), loss={
        'regression': regression_loss,
        'classification': classification_loss,
        'colors': colors_loss,
    },
        metrics={
            'colors':  'categorical_accuracy'
        }
     )

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    

    # NOTE: fit_generator is deprecated in TF2. Changed to fit().
    # start training
    return model.fit(
        train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )

    if args.wandb:
        wandb.save('checkpoints/**/*.h5')


if __name__ == '__main__':
    main()
