"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: May 28, 2022

Purpose: detect patch attacks
"""
import functools
import logging
import os

import tensorflow as tf

import custom_callbacks
import generator
import train_data_generator
import util
from tf2 import postprocess, efficientdet_keras
from util import get_victim_model

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttackDefender(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, initial_weights=None, config_override=None,
                 visualize_freq=200):
        super().__init__(name='Defender_Graph')
        self.model = model
        for layer in model.layers:
            layer.trainable = False
        self.config = self.model.config
        if config_override:
            self.model.config.override(config_override)
        self._antipatch = generator.define_generator(self.config.image_size)

        if initial_weights is not None:
            self._antipatch.load_weights(initial_weights)

        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self.cur_step = None
        self.tb = None
        self._trainable_variables = self._antipatch.trainable_variables

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self._antipatch.compile(*args, **kwargs)

    def filter_valid_boxes(self, images, boxes, scores):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        cond1 = tf.logical_and(tf.less_equal(boxes_w / w, 1.), tf.less_equal(boxes_h / h, 1.))
        cond2 = tf.logical_and(tf.greater(boxes_area, tf.constant(100.)), tf.greater_equal(scores, tf.constant(.3)))
        return tf.logical_and(cond1, cond2)

    def protege(self, images, training=False):
        with tf.name_scope('protege'):
            cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None, training=training)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            valid_boxes = self.filter_valid_boxes(images, boxes, scores)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
            classes = tf.ragged.boolean_mask(classes, valid_boxes)
        return boxes, scores, classes

    def _postprocessing(self, boxes, scores, classes):
        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
                return postprocess.nms(self.config, boxes[element], scores[element], classes[element], True)

            nms_boxes, nms_scores, nms_classes, nms_valid_len = tf.map_fn(single_batch_fn, tf.range(boxes.nrows()),
                                                                          fn_output_signature=(
                                                                              tf.TensorSpec((None, 4),
                                                                                            dtype=tf.float32),
                                                                              tf.TensorSpec((None,), dtype=tf.float32),
                                                                              tf.TensorSpec((None,), dtype=tf.float32),
                                                                              tf.TensorSpec((), dtype=tf.int32)))
            nms_boxes = postprocess.clip_boxes(nms_boxes, self.config.image_size)
            nms_boxes = tf.RaggedTensor.from_tensor(nms_boxes, lengths=nms_valid_len)
            nms_scores = tf.RaggedTensor.from_tensor(nms_scores, lengths=nms_valid_len)
        return nms_boxes, nms_scores

    def call(self, images, *, training=True):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        with tf.GradientTape() as tape:
            noises = self._antipatch(images, training=training)
            images = tf.clip_by_value(images + 2. * noises, -1., 1.)
            boxes_pred, scores_pred, classes_pred = self.protege(images, training=training)
            loss = 1. - tf.reduce_mean(scores_pred ** 2.) + tf.reduce_mean(tf.abs(noises)) + tf.reduce_mean(1. - noises ** 2.)

        self.add_metric(loss, name='loss')
        self.add_metric(tf.reduce_mean(scores_pred), name='mean_score')
        self.add_metric(tf.reduce_mean(noises ** 2.) ** .5, name='rms_noise')

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores_pred, classes_pred)
        func = functools.partial(self.vis_images, images, boxes_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._trainable_variables)

        return boxes_pred, scores_pred

    def vis_images(self, images, boxes_pred, training):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        tr = 'train' if training else 'val'

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        boxes_pred = convert_format(boxes_pred)
        images = tf.image.draw_bounding_boxes(images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.clip_by_value(images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        images = tf.cast(images, tf.uint8)

        with self.tb._writers[tr].as_default():
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(images)[0])

    def train_step(self, inputs):
        self.cur_step = self.tb._train_step
        grads = self(inputs)
        self.optimizer.apply_gradients([*zip(grads, self._trainable_variables)])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        self.cur_step = self.tb._val_step
        self(inputs, training=False)
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, dirpath, **kwargs):
        os.makedirs(dirpath)
        self._antipatch.save_weights(os.path.join(dirpath, 'antipatch.h5'))


def main(download_model=False):
    # obsolete, possibly non-functional code below
    from PIL import Image
    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    log_dir = util.ensure_empty_dir('log_dir')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # tf.config.run_functions_eagerly(True)

    victim_model = get_victim_model(download_model)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}, 'image_size': 300}
    model = PatchAttackDefender(victim_model, config_override=config_override, visualize_freq=200)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), run_eagerly=False)

    datasets: dict = train_data_generator.partition(model.config, 'downloaded_images', 'labels',
                                                    batch_size=1, shuffle=True)

    train_ds = datasets['train']['dataset']
    val_ds = datasets['val']['dataset']
    train_len = datasets['train']['length']
    val_len = datasets['val']['length']
    tb_callback = custom_callbacks.TensorboardCallback(log_dir, write_graph=True, write_steps_per_second=True)
    model.tb = tb_callback

    save_dir = util.ensure_empty_dir('save_dir')
    save_file = 'patch_{epoch:02d}_{val_loss:.4f}.h5'
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        steps_per_epoch=20,  # train_len,
                        validation_steps=20,  # val_len,
                        callbacks=[tb_callback,
                                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, save_file),
                                                                      monitor='val_loss',
                                                                      verbose=1,
                                                                      save_best_only=False,
                                                                      save_weights_only=True,
                                                                      mode='auto',
                                                                      save_freq='epoch',
                                                                      options=None,
                                                                      initial_value_threshold=None
                                                                      )
                                   ])
    patch = Image.fromarray(model.get_patch().astype('uint8'))
    patch.show('patch')


if __name__ == '__main__':
    main()
