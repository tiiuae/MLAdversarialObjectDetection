"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: attack the person detector
"""
import functools
import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import metrics
import custom_callbacks
import train_data_generator
import util
from tf2 import postprocess, efficientdet_keras, infer_lib

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, patch_loss_multiplier=1e-5, iou=.5,
                 min_patch_area=900, visualize_freq=200):
        super().__init__(name='Attacker_Graph')
        self.model = model
        self.config = self.model.config

        iou = iou
        self.model.config.override({'nms_configs': {'iou_thresh': iou, 'score_thresh': .5}})
        patch_img = (np.random.rand(256, 256, 3) * 255.).astype('uint8').astype(float)
        patch_img -= self.config.mean_rgb
        patch_img /= self.config.stddev_rgb
        # patch_img = np.ones((256, 256, 3), dtype=float) * self.config.mean_rgb
        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32,
                                  constraint=lambda x: tf.clip_by_value(x, 0., 1.))
        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self._patcher = Patcher(self._patch, name='Patcher')
        self._grad_processor = GradientProcessor(self._patch.shape, name='Gradient_Processor')
        self._images = None
        self._labels = None
        self.cur_train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.cur_test_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        tf.constant(1, tf.int64)
        self.cur_step = None
        self.tb = None

        patcher_scale = self._patcher.scale
        self._metric = metrics.AttackSuccessRate(min_patch_area / patcher_scale, iou_thresh=iou)
        self._patch_loss_multiplier = tf.constant(patch_loss_multiplier, tf.float32)

    def get_patch(self):
        return (self._patch.numpy() * self.config.stddev_rgb + self.config.mean_rgb).astype('uint8')

    def first_pass(self, images):
        boxes, scores, classes, _ = self.model(images, pre_mode=None)
        person_indices = tf.equal(classes, tf.constant(1.))
        boxes = tf.ragged.boolean_mask(boxes, person_indices)
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        valid_boxes = tf.greater(boxes_area, tf.constant(1000.))
        boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
        return boxes

    def second_pass(self, image):
        cls_outputs, box_outputs = self.model(image, pre_mode=None, post_mode=None)
        cls_outputs = postprocess.to_list(cls_outputs)
        box_outputs = postprocess.to_list(box_outputs)
        boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
        person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
        scores = tf.ragged.boolean_mask(scores, person_indices)

        boxes = tf.ragged.boolean_mask(boxes, person_indices)
        classes = tf.ragged.boolean_mask(classes, person_indices)
        return boxes, scores, classes

    def _postprocessing(self, boxes, scores, classes):

        def single_batch_fn(element):
            return postprocess.nms(self.config, element[0], element[1], element[2], True)

        nms_boxes, nms_scores, nms_classes, nms_valid_len = postprocess.batch_map_fn(
            single_batch_fn, [boxes, scores, classes])
        nms_boxes = postprocess.clip_boxes(nms_boxes, self.config.image_size)
        nms_boxes = tf.RaggedTensor.from_tensor(nms_boxes, lengths=nms_valid_len)
        nms_scores = tf.RaggedTensor.from_tensor(nms_scores, lengths=nms_valid_len)
        return nms_boxes, nms_scores

    def call(self, inputs, *, training=True):
        if isinstance(inputs, (tuple, list)):
            images, self._labels = inputs
        else:
            images, self._labels = inputs, None

        if self._labels is None:
            boxes = self.first_pass(images)
        else:
            boxes = self._labels

        if self._images is None:
            self._images = tf.Variable(images, name='inp_image', dtype=tf.float32)
        else:
            self._images.assign(images)

        patch_boxes, transform_decisions = self._patcher([boxes, self._images])

        with tf.GradientTape(persistent=True) as tape:
            boxes_pred, scores, classes = self.second_pass(self._images)
            loss = tf.reduce_max(scores)
            tv_loss = self.tv_loss()

        self.add_loss(loss)
        self.add_loss(tv_loss)

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores, classes)
        self.add_metric(self._metric(boxes_pred, boxes), name='mean_asr')

        func = functools.partial(self.vis_images, boxes_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            gradients = tape.gradient(loss, self._images)
            gradients = self._grad_processor([gradients, patch_boxes, transform_decisions])
            gradients = gradients + tape.gradient(tv_loss, self._patch)
            return gradients

        return boxes_pred, scores_pred

    def vis_images(self, boxes_pred, training):
        images, labels = self._images, self._labels
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        boxes_pred = convert_format(boxes_pred)
        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))
        images = tf.image.draw_bounding_boxes(images, boxes_pred, tf.constant([[0., 0., 1.]]))
        tr = 'train' if training else 'val'
        with self.tb._writers[tr].as_default():
            if training:
                tf.summary.image('Current patch', self._patch[tf.newaxis], step=self.cur_step)
            tf.summary.image('Sample', images, step=self.cur_step)

    def tv_loss(self):
        """TV loss"""
        pixel_front = tf.concat([self._patch[:, 1:], self._patch[:, -1:]], axis=1)
        pixel_down = tf.concat([self._patch[1:, :], self._patch[-1:, :]], axis=0)
        return tf.reduce_sum(((self._patch - pixel_front) ** 2. +
                              (self._patch - pixel_down) ** 2.) ** .5) * self._patch_loss_multiplier

    def train_step(self, inputs):
        self.cur_step = self.cur_train_step
        self.cur_step.assign_add(tf.constant(1, tf.int64))
        self.reset_state()
        grads = self(inputs)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        self.optimizer.apply_gradients([(grads, self._patch)])
        ret = {'loss': self.losses[0], 'tv_loss': self.losses[1]}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def test_step(self, inputs):
        self.cur_step = self.cur_test_step
        self.cur_step.assign_add(tf.constant(1, tf.int64))
        self.reset_state()
        self(inputs, training=False)
        ret = {'loss': self.losses[0], 'tv_loss': self.losses[1]}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def reset_state(self):
        self._labels = None

    def save_weights(self, filepath, **kwargs):
        plt.imsave(filepath, self.get_patch())


class GradientProcessor(tf.keras.layers.Layer):
    def __init__(self, patch_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_shape = patch_shape
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._agg = tf.Variable(tf.zeros(patch_shape, dtype=tf.float32), trainable=False)
        self._patch_boxes = None
        self._transform_decisions = None
        self._gradients = None

    def inner_loop(self, _):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(self._patch_boxes[self._batch_counter,
                                                                                        self._patch_counter], tf.int32),
                                                              4)
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        gradients = self._gradients[self._batch_counter, ymin_patch:ymax, xmin_patch:xmax]
        transform_decisions = self._transform_decisions[self._batch_counter, self._patch_counter]
        # gradients = tf.cond(transform_decisions[2], lambda: tf.image.flip_up_down(gradients), lambda: gradients)
        # gradients = tf.cond(transform_decisions[1], lambda: tf.image.flip_left_right(gradients), lambda: gradients)
        # gradients = tf.cond(transform_decisions[0], lambda: tf.image.rot90(gradients, k=3), lambda: gradients)
        gradients = tf.image.resize(gradients, self._patch_shape[:-1])
        self._agg.assign_add(gradients)
        self._patch_counter.assign_add(tf.constant(1))
        return [self._patch_counter]

    def batch_loop(self, _):
        self._patch_counter.assign(tf.constant(0))
        tf.while_loop(lambda _: tf.less(self._patch_counter, tf.cast(self._patch_boxes[self._batch_counter].nrows(),
                                                                     tf.int32)),
                      self.inner_loop, [self._patch_counter])
        self._batch_counter.assign_add(tf.constant(1))
        return [self._batch_counter]

    def call(self, inputs, *args, **kwargs):
        self._gradients, self._patch_boxes, self._transform_decisions = inputs
        self._batch_counter.assign(tf.constant(0))
        self._agg.assign(tf.zeros(self._patch_shape, dtype=tf.float32))
        tf.while_loop(lambda _: tf.less(self._batch_counter, tf.cast(self._patch_boxes.nrows(), tf.int32)),
                      self.batch_loop, [self._batch_counter])
        return self._agg


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch: tf.Variable, *args, aspect=1., origin=(.5, .5), scale=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._images = None
        self.aspect = aspect
        self.origin = origin
        self.scale = scale

    def add_patches_to_image(self, boxes):
        self._patch_counter.assign(tf.constant(0))
        patch_boxes = tf.vectorized_map(self.create, boxes)
        patch_boxes = tf.gather_nd(patch_boxes,
                                   tf.where(tf.greater(tf.cast(patch_boxes[:, 2] * patch_boxes[:, 3], tf.int32),
                                                       tf.constant(900))))
        transform_decisions = tf.cast(tf.random.uniform(shape=(tf.shape(patch_boxes)[0], 3), minval=0, maxval=2,
                                                        dtype=tf.int32), tf.bool)
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes)
        tf.while_loop(lambda _, i: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]),
                      loop_fn, [self._patch_counter, transform_decisions])
        self._batch_counter.assign_add(tf.constant(1))
        return tf.RaggedTensor.from_tensor(tf.concat([patch_boxes, tf.cast(transform_decisions, tf.float32)], axis=1),
                                           row_splits_dtype=tf.int32)

    def add_patch_to_image(self, patch_boxes, _, transform_decisions):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        im = tf.image.resize(self._patch, tf.stack([patch_h, patch_w]))
        # im = tf.cond(transform_decisions[self._patch_counter, 0], lambda: tf.image.rot90(im), lambda: im)
        # im = tf.cond(transform_decisions[self._patch_counter, 1], lambda: tf.image.rot90(im), lambda: im)
        # im = tf.cond(transform_decisions[self._patch_counter, 2], lambda: tf.image.rot90(im), lambda: im)

        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        self._images[self._batch_counter, ymin_patch:ymax, xmin_patch:xmax].assign(im)
        self._patch_counter.assign_add(tf.constant(1))
        return [self._patch_counter, transform_decisions]

    def create(self, bbox):
        ymin, xmin, ymax, xmax = tf.unstack(bbox, 4)

        h = ymax - ymin
        w = xmax - xmin

        patch_h = h * self.scale
        patch_w = self.aspect * patch_h

        ymin_patch = ymin + h * self.origin[1]
        xmin_patch = xmin + w * self.origin[0]

        shape = tf.cast(tf.shape(self._images), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + patch_h, shape[1]),
                             lambda: shape[1] - patch_h, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + patch_w, shape[2]),
                             lambda: shape[2] - patch_w, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w])

    def call(self, inputs, *args, **kwargs):
        batch_boxes, self._images = inputs
        self._batch_counter.assign(tf.constant(0))
        result = tf.map_fn(self.add_patches_to_image, batch_boxes,
                           fn_output_signature=tf.RaggedTensorSpec((None, 7), tf.float32, 1, tf.int32))
        patch_boxes = result[:, :, :4]
        transform_decisions = tf.cast(result[:, :, 4:], tf.bool)
        return patch_boxes, transform_decisions


def ensure_empty_dir(dirname):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        shutil.rmtree(dirname, ignore_errors=True)
        os.makedirs(dirname)
    return dirname


def main(download_model=False):
    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    if download_model:
        # Download checkpoint.
        util.download(MODEL)
        logger.info(f'Using model in {MODEL}')

    log_dir = ensure_empty_dir('log_dir')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL)
    model = PatchAttacker(driver.model)
    model.compile(optimizer='adam', run_eagerly=False)

    train_ds, val_ds, test_ds = train_data_generator.partition(model.config, 'downloaded_images', 'labels',
                                                               batch_size=1, shuffle=False)

    tb_callback = custom_callbacks.TensorboardCallback(log_dir, write_graph=True)
    model.tb = tb_callback

    save_dir = ensure_empty_dir('save_dir')
    save_file = 'patch_{epoch:02d}.png'
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, steps_per_epoch=2,  # len(data_gen),
                        validation_steps=2,
                        callbacks=[tb_callback,
                                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, save_file),
                                                                      monitor='val_loss',
                                                                      verbose=1,
                                                                      save_best_only=True,
                                                                      save_weights_only=True,
                                                                      mode='auto',
                                                                      save_freq='epoch',
                                                                      options=None,
                                                                      initial_value_threshold=None,
                                                                      )
                                   ])
    patch = model.get_patch()
    plt.imshow(patch)
    plt.show()


if __name__ == '__main__':
    main()
