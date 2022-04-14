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
            tf.summary.scalar('loss', logs['score_loss'], step=batch)
            tf.summary.scalar('tv_loss', logs['tv_loss'], step=batch)
            tf.summary.scalar('recall', self.model.metrics[0].result(), step=batch)
            if not batch % 1:
                rand_image, patch = self.model.get_vis_images()
                tf.summary.image('Current patch', patch, step=batch)
                tf.summary.image('Sample', rand_image, step=batch)
