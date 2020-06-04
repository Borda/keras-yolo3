"""
Retrain the YOLO model for your own dataset.

>> python train_bottleneck.py \
       --path_dataset ./model_data/VOC_2007_train.txt \
       --path_weights ./model_data/tiny-yolo.h5 \
       --path_anchors ./model_data/tiny-yolo_anchors.csv \
       --path_classes ./model_data/coco_classes.txt \
       --path_output ./model_data
"""

import os
import sys
import time
import logging
from functools import partial

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.model import create_model_bottleneck
from keras_yolo3.utils import get_anchors, get_nb_classes, data_generator, generator_bottleneck, get_dataset_class_names
from scripts.training import parse_params, load_config, load_training_lines, _export_classes, _export_model


DEFAULT_CONFIG = {
    'image-size': (608, 608),
    'batch-size':
        {'head': 16, 'bottlenecks': 8, 'full': 16},
    'epochs':
        {'head': 50, 'bottlenecks': 30, 'full': 50},
    'valid-split': 0.1,
    'recompute-bottlenecks': True,
    'generator': {}
}
NAME_BOTTLENECKS = 'bottlenecks.npz'
NAME_CHECKPOINT = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'


def _main(path_dataset, path_anchors, path_weights=None, path_output='.',
          path_config=None, path_classes=None, nb_gpu=1, **kwargs):

    config = load_config(path_config, DEFAULT_CONFIG)
    anchors = get_anchors(path_anchors)
    nb_classes = get_nb_classes(path_dataset)
    logging.info('Using %i classes', nb_classes)
    _export_classes(get_dataset_class_names(path_dataset, path_classes), path_output)

    # make sure you know what you freeze
    model, bottleneck_model, last_layer_model = create_model_bottleneck(
        config['image-size'], anchors, nb_classes, freeze_body=2,
        weights_path=path_weights, nb_gpu=nb_gpu)
    # if create blank use image-size, else take loaded from model file
    config['image-size'] = model._input_layers[0].input_shape[1:3]

    log_tb = TensorBoard(log_dir=path_output)
    checkpoint = ModelCheckpoint(os.path.join(path_output, NAME_CHECKPOINT),
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    lines_train, lines_valid, num_val, num_train = load_training_lines(path_dataset,
                                                                       config['valid-split'])

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    _yolo_loss = lambda y_true, y_pred: y_pred  # use custom yolo_loss Lambda layer.
    _data_gene_bottleneck = partial(generator_bottleneck,
                                    batch_size=config['batch-size']['bottlenecks'],
                                    input_shape=config['image-size'],
                                    anchors=anchors,
                                    nb_classes=nb_classes,
                                    **config['generator'])
    _data_generator = partial(data_generator,
                              input_shape=config['image-size'],
                              anchors=anchors,
                              nb_classes=nb_classes,
                              **config['generator'])

    epochs_head = config['epochs'].get('head', 0)
    epochs_btnc = config['epochs'].get('bottlenecks', 0)
    if epochs_btnc > 0 or epochs_head > 0:
        # perform bottleneck training
        path_bottlenecks = os.path.join(path_output, NAME_BOTTLENECKS)
        if not os.path.isfile(path_bottlenecks) or config['recompute-bottlenecks']:
            logging.info('calculating bottlenecks')
            bottlenecks = bottleneck_model.predict_generator(
                _data_generator(lines_train + lines_valid,
                                randomize=False, batch_size=config['batch-size']['bottlenecks']),
                steps=(len(lines_train + lines_valid) // config['batch-size']['bottlenecks']) + 1,
                max_queue_size=1)
            np.savez(path_bottlenecks, bot0=bottlenecks[0], bot1=bottlenecks[1], bot2=bottlenecks[2])

        # load bottleneck features from file
        dict_bot = np.load(path_bottlenecks)
        bottlenecks_train = [dict_bot[bot_][:num_train] for bot_ in ("bot0", "bot1", "bot2")]
        bottlenecks_val = [dict_bot[bot_][num_train:] for bot_ in ("bot0", "bot1", "bot2")]

        # train last layers with fixed bottleneck features
        logging.info('Training last layers with bottleneck features '
                     'with %i samples, val on %i samples and batch size %i.',
                     num_train, num_val, config['batch-size']['bottlenecks'])
        last_layer_model.compile(optimizer='adam', loss={'yolo_loss': _yolo_loss})
        t_start = time.time()
        last_layer_model.fit_generator(
            _data_gene_bottleneck(lines_train, bottlenecks=bottlenecks_train),
            steps_per_epoch=max(1, num_train // config['batch-size']['bottlenecks']),
            validation_data=_data_gene_bottleneck(lines_valid, bottlenecks=bottlenecks_val),
            validation_steps=max(1, num_val // config['batch-size']['bottlenecks']),
            epochs=epochs_btnc,
            initial_epoch=0,
            max_queue_size=1)
        _export_model(model, path_output, '', '_bottleneck')

        # train last layers with random augmented data
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': _yolo_loss})  # use custom yolo_loss Lambda layer.
        logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                     num_train, num_val, config['batch-size']['head'])
        t_start = time.time()
        model.fit_generator(
            _data_generator(lines_train, batch_size=config['batch-size']['head']),
            steps_per_epoch=max(1, num_train // config['batch-size']['head']),
            validation_data=_data_generator(lines_valid, batch_size=config['batch-size']['head']),
            validation_steps=max(1, num_val // config['batch-size']['head']),
            epochs=epochs_btnc + epochs_head,
            initial_epoch=epochs_btnc,
            callbacks=[log_tb, checkpoint])
        logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
        _export_model(model, path_output, '', '_head')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if config['epochs'].get('full', 0) > 0:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        logging.info('Unfreeze all of the layers.')

        # note that more GPU memory is required after unfreezing the body
        logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                     num_train, num_val, config['batch-size']['full'])
        t_start = time.time()
        model.fit_generator(
            _data_generator(lines_train, batch_size=config['batch-size']['full']),
            steps_per_epoch=max(1, num_train // config['batch-size']['full']),
            validation_data=_data_generator(lines_valid, batch_size=config['batch-size']['full']),
            validation_steps=max(1, num_val // config['batch-size']['full']),
            epochs=epochs_btnc + epochs_head + config['epochs']['full'],
            initial_epoch=epochs_btnc + epochs_head,
            callbacks=[log_tb, checkpoint, reduce_lr, early_stopping])
        logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
        _export_model(model, path_output, '', '_full')

    # Further training if needed.


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    a_params = parse_params()
    _main(**a_params)
    logging.info('Done')
