"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 7, 2022

Purpose: detect patch attacks
"""
import ast
import functools
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import tfplot
from tifffile import tifffile

import generator
import histogram_matcher
import hparams_config as hparams
import util
from tf2 import postprocess, efficientdet_keras

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttackDefender(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, initial_weights=None, eval_patch_weights=None,
                 config_override=None, visualize_freq=200):
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

        patch = tifffile.imread(os.path.join(eval_patch_weights, 'patch.tiff'))
        with open(os.path.join(eval_patch_weights, 'scale.txt')) as f:
            scale = ast.literal_eval(f.read())

        min_patch_area = 4

        self._masker = Masker(patch, scale, min_patch_area=min_patch_area, name='Masker')

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
        cond2 = tf.logical_and(tf.greater(boxes_area, tf.constant(100.)),
                               tf.greater_equal(scores, self.config.nms_configs.score_thresh))
        return tf.logical_and(cond1, cond2)

    def odet_model(self, images, score_thresh=None):
        if score_thresh is not None:
            config = hparams.Config(self.config.as_dict())
            config.nms_configs.score_thresh = score_thresh
            assert config.nms_configs.score_thresh == score_thresh
        else:
            config = self.config

        with tf.name_scope('first_pass'):
            cls_outputs, box_outputs = self.model(images, pre_mode=None, post_mode=None)
            cls_outputs = postprocess.to_list(cls_outputs)
            box_outputs = postprocess.to_list(box_outputs)
            boxes, scores, classes = postprocess.pre_nms(config.as_dict(), cls_outputs, box_outputs)
            person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
            scores = tf.ragged.boolean_mask(scores, person_indices)

            boxes = tf.ragged.boolean_mask(boxes, person_indices)
            classes = tf.ragged.boolean_mask(classes, person_indices)

            boxes, scores = self._postprocessing(boxes, scores, classes, score_thresh=score_thresh)

            valid_boxes = self.filter_valid_boxes(images, boxes, scores)
            boxes = tf.ragged.boolean_mask(boxes, valid_boxes)
            scores = tf.ragged.boolean_mask(scores, valid_boxes)
        return boxes, scores

    def _postprocessing(self, boxes, scores, classes, score_thresh=None):
        if score_thresh is not None:
            config = hparams.Config(self.config.as_dict())
            config.nms_configs.score_thresh = score_thresh
            assert config.nms_configs.score_thresh == score_thresh
        else:
            config = self.config

        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
                return postprocess.nms(config, boxes[element], scores[element], classes[element], True)

            if not isinstance(boxes, tf.RaggedTensor):
                boxes = tf.RaggedTensor.from_tensor(boxes)
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
        boxes, scores = self.odet_model(images)

        with tf.GradientTape() as tape:
            images, targets = self._masker([boxes, images], training=training)

            if not training:
                boxes, scores = self.odet_model(images, score_thresh=0.)

            updates = 2. * self._antipatch(images, training=training)
            flat_targets = tf.reshape(targets, (tf.shape(images)[0], -1))
            flat_updates = tf.reshape(updates, (tf.shape(images)[0], -1))
            loss = tf.reduce_sum(tf.reduce_mean((flat_targets - flat_updates) ** 2., axis=1))

        self.add_metric(loss, name='loss')

        func = functools.partial(self.vis_images, images, updates, boxes, scores, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._trainable_variables)

    @staticmethod
    @tfplot.autowrap(figsize=(4, 4))
    def plot_scores(x, y, step, *, ax):

        def make_df(arr, label):
            df = pd.DataFrame(arr, columns=['scores'])
            df['label'] = label
            df[''] = ''  # seaborn issue
            return df

        df1 = make_df(x, 'original')
        df2 = make_df(y, 'recovered')
        df = pd.concat([df1, df2], ignore_index=True)
        ax = sns.violinplot(x='', y='scores', hue='label', split=True, data=df, ax=ax)
        ax.legend()

    def vis_images(self, images, updates, labels, scores, training):
        _, h, w, _ = tf.unstack(tf.cast(tf.shape(images), tf.float32))

        tr = 'train' if training else 'val'

        def convert_format(box):
            ymin, xmin, ymax, xmax = tf.unstack(box.to_tensor(), axis=2)
            return tf.stack([ymin / h, xmin / w, ymax / h, xmax / w], axis=2)

        labels = convert_format(labels)
        updated_images = tf.clip_by_value(images + updates, -1., 1.)
        boxes_pred, scores_pred = self.odet_model(updated_images, score_thresh=0.)
        boxes_pred = convert_format(boxes_pred)

        images = tf.image.draw_bounding_boxes(images, labels, tf.constant([[0., 1., 0.]]))

        updated_images = tf.image.draw_bounding_boxes(updated_images, boxes_pred, tf.constant([[0., 0., 1.]]))
        images = tf.clip_by_value(images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        images = tf.cast(images, tf.uint8)
        updated_images = tf.clip_by_value(updated_images * self.config.stddev_rgb + self.config.mean_rgb, 0., 255.)
        updated_images = tf.cast(updated_images, tf.uint8)
        images = tf.reshape(tf.concat([images, updated_images], axis=1), (-1, *updated_images[0].shape))

        scores = scores.merge_dims(0, -1)
        scores_pred = scores_pred.merge_dims(0, -1)
        plot = self.plot_scores(scores, scores_pred, self.cur_step)

        with self.tb._writers[tr].as_default():
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(images)[0])
            tf.summary.image('Scores', plot[tf.newaxis], step=self.cur_step)

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


class Masker(tf.keras.layers.Layer):
    def __init__(self, patch: tf.keras.Model, scale_regressor, *args, min_patch_area=60, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self._patch = patch
        self.min_patch_area = min_patch_area
        self._matcher = histogram_matcher.BrightnessMatcher(name='Brightness_Matcher')
        self._scale = scale_regressor
        self.is_training = False
        self._train_patches = None

    def random_print_adjust(self, patch):
        w = tf.random.normal((1, 1, 3), .5, .1)
        b = tf.random.normal((1, 1, 3), 0., .01)
        return tf.clip_by_value(w * patch + b, -1., 1.)

    def add_patches_to_image(self, image):
        h, w, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        boxes = self._boxes[self._batch_counter]

        patch_boxes = tf.vectorized_map(functools.partial(self.create, image), boxes)
        patch_boxes = tf.reshape(patch_boxes, shape=(-1, 5))
        valid_indices = tf.where(tf.greater(patch_boxes[:, 2] * patch_boxes[:, 3],
                                            tf.constant(self.min_patch_area, tf.float32)))
        patch_boxes = tf.gather_nd(patch_boxes, valid_indices)

        if not self.is_training:
            patch = self.random_print_adjust(self._patch)
            patch = self._matcher((patch, image))
        else:
            patch = self._train_patches[self._batch_counter]
            patch = self.random_print_adjust(patch)
            patch = self._matcher((patch, image))

        self._patch_counter.assign(tf.constant(0))
        mask = tf.zeros_like(image)
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes, patch, image)
        image, mask, _ = tf.while_loop(lambda image, mask, j: tf.less(self._patch_counter, tf.shape(boxes)[0]),
                                       loop_fn, [image, mask, self._patch_counter])

        self._batch_counter.assign_add(tf.constant(1))
        return image, mask

    def add_patch_to_image(self, patch_boxes, patch, oimage, image, mask, j):
        ymin_patch, xmin_patch, patch_h, patch_w, diag = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        ymax_patch = ymin_patch + diag
        xmax_patch = xmin_patch + diag
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax_patch), tf.range(xmin_patch, xmax_patch), indexing='ij'),
                       axis=-1)

        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]), antialias=True)
        im += tf.random.uniform(shape=tf.shape(im), minval=-.1, maxval=.1)
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
        im = tf.where(tf.less(im, -1.), patch_bg, im)
        im = tf.clip_by_value(im, -1., 1.)

        image = tf.tensor_scatter_nd_update(image, idx, im)
        patch_bg = oimage[ymin_patch: ymax_patch, xmin_patch: xmax_patch]
        mask = tf.tensor_scatter_nd_update(mask, idx, patch_bg - im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, mask, self._patch_counter]

    def create(self, image, item):
        ymin, xmin, ymax, xmax = tf.unstack(item, 4)

        h = ymax - ymin
        w = xmax - xmin

        longer_side = tf.maximum(h, w)

        if self.is_training:
            tolerance = .5
            scale = tf.random.uniform((), minval=.3, maxval=.5)
        else:
            tolerance = 0.
            scale = self._scale

        patch_size = tf.floor(longer_side * scale)
        diag = tf.minimum((2. ** .5) * patch_size, tf.cast(image.shape[1], tf.float32))

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

    def call(self, inputs, training=False):
        self._boxes, images = inputs
        self.is_training = training
        if training:
            patches = tf.random.shuffle(images[:, :240, :240, :])
            patches = tf.image.random_flip_left_right(patches)
            patches = tf.image.random_flip_up_down(patches)
            patches = tf.image.adjust_contrast(patches, 2.)
            patches = tf.image.adjust_saturation(patches, 2.)
            self._train_patches = patches
        else:
            self._train_patches = None
        self._batch_counter.assign(tf.constant(0))
        return tf.map_fn(self.add_patches_to_image, images,
                         fn_output_signature=(tf.TensorSpec(dtype=tf.float32, shape=images[0].shape),
                                              tf.TensorSpec(dtype=tf.float32, shape=images[0].shape)))
