from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from .network import VAE_types

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import Iterator
from tensorflow.python import debug as tf_debug



def train(adata, network, output_dir="E://work_code//Dva//output", optimizer='RMSprop', learning_rate=None,
          epochs=100, reduce_lr=10, output_subset=None, use_raw_as_output=True,
          early_stop=50, batch_size=32, clip_grad=5., save_weights=False,
          validation_split=0.1, tensorboard=False, verbose=True, threads=None,
          **kwds):

    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=threads,
                inter_op_parallelism_threads=threads,
            )
        )
    )
    model = network.model
    loss = network.loss
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)

    if loss is not None:
        model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose: model.summary()

    # 获得数据
    train_data = adata.values
    inputs = train_data
    rowName = adata.index
    output = adata.values

    loss = model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose,
                     **kwds)

    # print("loss1:", loss.history['loss'])

    return loss