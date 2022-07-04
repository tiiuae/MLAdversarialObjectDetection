"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: attack the person detector with dynamic patches
"""
import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tfplot

import dataloader
import generator
from tf2 import postprocess, efficientdet_keras


class DynamicPatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, initial_weights=None, config_override=None,
                 visualize_freq=200):
        super().__init__(name='Attacker_Graph')
        self.model = model
        self.config = self.model.config
        if config_override:
            self.model.config.override(config_override)
        self._patch = generator.define_generator(128, generator.PatchGenerator)

        if initial_weights is not None:
            self._patch.load_weights(initial_weights)

        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self._patcher = Patcher(self._patch, name='Patcher')
        self.cur_step = None
        self.tb = None
        self._trainable_variables = self._patch.trainable_variables

        self.bins = np.arange(self.config.nms_configs.score_thresh, .805, .01, dtype='float32')
        self.asr = [tf.Variable(0., dtype=tf.float32, trainable=False) for _ in self.bins]

    def filter_valid_boxes(self, images, boxes, scores, thresh=True):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))
        boxes_h = boxes[:, :, 2] - boxes[:, :, 0]
        boxes_w = boxes[:, :, 3] - boxes[:, :, 1]
        boxes_area = boxes_h * boxes_w
        cond1 = tf.logical_and(tf.less_equal(boxes_w / w, 1.), tf.less_equal(boxes_h / h, 1.))
        if thresh:
            cond2 = tf.logical_and(tf.greater(boxes_area, tf.constant(100.)),
                                   tf.greater_equal(scores, self.config.nms_configs.score_thresh))
        else:
            cond2 = tf.greater(boxes_area, tf.constant(100.))
        return tf.logical_and(cond1, cond2)

    def first_pass(self, images):
        cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)
        with tf.name_scope('first_pass'):
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

            boxes, scores = self._postprocessing(boxes, scores, classes)

        return boxes, scores

    def second_pass(self, images):
        cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)
        with tf.name_scope('attack_pass'):
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            valid_boxes = self.filter_valid_boxes(images, boxes, scores, thresh=False)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
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
        boxes, scores = self.first_pass(images)

        with tf.GradientTape() as tape:
            images, mean_scales = self._patcher([boxes, images])
            boxes_pred, scores_pred, classes = self.second_pass(images)
            max_scores = tf.maximum(tf.reduce_max(scores_pred, axis=1), 0.)
            scale_losses = (max_scores - mean_scales) ** 2.
            loss = tf.reduce_sum(max_scores ** 2. + scale_losses)

        self.add_metric(loss, name='loss')
        self.add_metric(tf.reduce_mean(mean_scales), name='mean_scale')
        self.add_metric(tf.reduce_sum(scale_losses), name='scale_loss')
        self.add_metric(tf.reduce_mean(max_scores), name='mean_max_score')
        self.add_metric(tf.math.reduce_std(max_scores), name='std_max_score')

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores_pred, classes)
        asr = self.calc_asr(scores, scores_pred, boxes, boxes_pred)
        self.add_metric(asr, name='asr')
        self.add_metric(asr / tf.reduce_mean(mean_scales), name='asr_to_mean_scale')

        func = functools.partial(self.vis_images, images, boxes, scores, boxes_pred, scores_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._trainable_variables)

        return boxes_pred, scores_pred

    @staticmethod
    @tfplot.autowrap(figsize=(4, 4))
    def plot_asr(x: np.ndarray, y: np.ndarray, step, *, ax, color='blue'):
        ax.plot(x, y, color=color)
        ax.set_ylim(0., 1.)
        ax.set_xlabel('score_thresh')
        ax.set_ylabel('attack_success_rate')

    def calc_asr(self, scores, scores_pred, boxes, boxes_pred, *, score_thresh=.5):
        filt = tf.greater_equal(scores, tf.constant(score_thresh))
        labels_filt = tf.ragged.boolean_mask(boxes, filt)

        filt = tf.greater_equal(scores_pred, tf.constant(score_thresh))
        boxes_pred_filt = tf.ragged.boolean_mask(boxes_pred, filt)
        return 1. - tf.cast(tf.size(boxes_pred_filt.flat_values),
                            tf.float32) / (tf.cast(tf.size(labels_filt.flat_values), tf.float32) +
                                           tf.keras.backend.epsilon())
        # return self._metric(boxes_pred_filt, labels_filt)

    def vis_images(self, images, labels, scores, boxes_pred, scores_pred, training):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        tr = 'train' if training else 'val'

        for i, score in enumerate(self.bins):
            self.asr[i].assign(self.calc_asr(scores, scores_pred, labels, boxes_pred, score_thresh=score))
        plot = self.plot_asr(self.bins, self.asr, self.cur_step)

        with self.tb._writers[tr].as_default():
            tf.summary.image('ASR', plot[tf.newaxis], step=self.cur_step)

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        boxes_pred = convert_format(boxes_pred)
        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))
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
        self._patch.save_weights(os.path.join(dirpath, 'patch.h5'))


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch, *args, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self._patch = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self.loss = None

    def add_patches_to_image(self, image):
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]

        self._patch_counter.assign(tf.constant(0))
        loop_fn = functools.partial(self.add_patch_to_image, boxes)
        image, _ = tf.while_loop(lambda image, j: tf.less(self._patch_counter, tf.shape(boxes)[0]),
                                 loop_fn, [image, self._patch_counter])

        self.loss[self._batch_counter].assign(tf.math.divide_no_nan(self.loss[self._batch_counter],
                                                                    tf.cast(self._patch_counter.value(), tf.float32)))
        self._batch_counter.assign_add(tf.constant(1))
        return image

    def add_patch_to_image(self, boxes, image, j):
        ymin, xmin, ymax, xmax = tf.unstack(tf.cast(boxes[self._patch_counter], tf.int32))

        patch_bg = image[ymin: ymax, xmin: xmax]
        patch_bg = self.preprocessing(patch_bg)
        im, scale = self._patch(patch_bg[tf.newaxis])
        im, scale = tf.squeeze(im), tf.squeeze(scale)
        im = self.random_print_adjust(im)
        self.loss[self._batch_counter].assign(scale + self.loss[self._batch_counter])

        patch_coords = tf.squeeze(self.create(image, scale, boxes[self._patch_counter]))
        image = self.add_patch_to_image_helper(patch_coords, im, image)

        self._patch_counter.assign_add(tf.constant(1))
        return [image, self._patch_counter]

    def random_print_adjust(self, patch):
        w = tf.random.normal((1, 1, 3), .5, .1)
        b = tf.random.normal((1, 1, 3), 0., .01)
        return tf.clip_by_value(w * patch + b, -1., 1.)

    def preprocessing(self, image):
        input_processor = dataloader.DetectionInputProcessor(image, 128)
        input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        return image

    def add_patch_to_image_helper(self, box, patch, image):
        ymin_patch, xmin_patch, patch_h, patch_w, diag = tf.unstack(tf.cast(box, tf.int32))
        ymax_patch = ymin_patch + diag
        xmax_patch = xmin_patch + diag
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax_patch), tf.range(xmin_patch, xmax_patch), indexing='ij'),
                       axis=-1)

        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]), antialias=True)
        im += tf.random.uniform(shape=tf.shape(im), minval=-.01, maxval=.01)
        im = tf.image.random_brightness(im, .3)
        im = tf.clip_by_value(im, -1., 1.)

        offset = (diag - patch_h) / 2
        top = left = tf.cast(tf.math.floor(offset), tf.int32)
        bottom = right = tf.cast(tf.math.ceil(offset), tf.int32)
        pads = tf.reshape(tf.stack([top, bottom, left, right, 0, 0]), (-1, 2))
        im = tf.pad(im, pads, constant_values=-2)
        angle = tf.random.uniform(shape=(), minval=-20. * np.pi / 180., maxval=20. * np.pi / 180.)
        im = tfa.image.rotate(im, angle, interpolation='bilinear', fill_value=-2.)
        patch_bg = image[ymin_patch: ymax_patch, xmin_patch: xmax_patch]
        # tf.print(tf.shape(im), tf.shape(patch_bg))
        im = tf.where(tf.less(im, -1.), patch_bg, im)
        im = tf.clip_by_value(im, -1., 1.)

        image = tf.tensor_scatter_nd_update(image, idx, im)
        return image

    def create(self, image, scale, item):
        ymin, xmin, ymax, xmax = tf.unstack(item, 4)

        h = ymax - ymin
        w = xmax - xmin

        longer_side = tf.maximum(h, w)
        tolerance = .2

        patch_size = tf.floor(longer_side * scale)
        diag = tf.minimum((2. ** .5) * patch_size, tf.cast(image.shape[1], tf.float32))
        # tf.print(patch_size)

        orig_y = ymin + h / 2. + tf.random.uniform((), minval=-tolerance * h / 2., maxval=tolerance * h / 2.)
        orig_x = xmin + w / 2. + tf.random.uniform((), minval=-tolerance * w / 2., maxval=tolerance * w / 2.)

        patch_w = patch_size
        patch_h = patch_size

        ymin_patch = tf.maximum(orig_y - diag / 2., 0.)
        xmin_patch = tf.maximum(orig_x - diag / 2., 0.)

        shape = tf.cast(tf.shape(image), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + diag, shape[0]),
                             lambda: shape[0] - diag, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + diag, shape[1]),
                             lambda: shape[1] - diag, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w, diag])

    def call(self, inputs):
        self._boxes, images = inputs
        zeros = tf.zeros(tf.shape(images)[0], dtype=tf.float32)
        if self.loss is None:
            self.loss = tf.Variable(zeros, trainable=False)
        else:
            self.loss.assign(zeros)
        self._batch_counter.assign(tf.constant(0))
        return tf.map_fn(self.add_patches_to_image, images), self.loss.value()
