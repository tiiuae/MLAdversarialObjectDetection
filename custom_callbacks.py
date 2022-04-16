"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 14, 2022

Purpose: Tensorboard based visualization
"""
import tensorflow as tf


class TensorboardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_batch_end(self, batch, logs=None):
        with self._writers['train'].as_default():
            for k, v in logs.items():
                tf.summary.scalar(k, v, step=batch)

    def on_test_batch_end(self, batch, logs=None):
        with self._writers['val'].as_default():
            for k, v in logs.items():
                tf.summary.scalar(f'val_{k}', v, step=batch)
