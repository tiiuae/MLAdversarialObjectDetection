"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: attack the person detector
"""
import functools
import logging
import os

import numpy as np
import tensorflow as tf
import tifffile

import custom_callbacks
import histogram_matcher
import metrics
import train_data_generator
import util
from tf2 import postprocess, efficientdet_keras, infer_lib

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, starting_patch=None, patch_loss_multiplier=1e-2,
                 min_patch_height=1, config_override=None, visualize_freq=200):
        super().__init__(name='Attacker_Graph')
        self.model = model
        self.config = self.model.config
        if config_override:
            self.model.config.override(config_override)
        if starting_patch is None:
            patch_img = (np.random.rand(512, 512, 3) * 255.).astype('uint8').astype(float)
        else:
            patch_img = tifffile.imread(starting_patch)
        patch_img -= self.config.mean_rgb
        patch_img /= self.config.stddev_rgb
        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32,
                                  constraint=lambda x: tf.clip_by_value(x, -1., 1.))
        self.visualize_freq = tf.constant(visualize_freq, tf.int64)
        self._denorm = Denormalizer(self.config, name='Denormalizer')
        self._histmatch = histogram_matcher.HistogramMatcher(name='Hist_Matcher')
        self._patcher = Patcher(self._patch, self._histmatch, min_patch_height=min_patch_height, name='Patcher')
        self._images = None
        self._labels = None
        tf.constant(1, tf.int64)
        self.cur_step = None
        self.tb = None

        patcher_scale = self._patcher.scale
        iou = self.config.nms_configs.iou_thresh
        self._metric = metrics.AttackSuccessRate(min_patch_height / patcher_scale, iou_thresh=iou)
        self._patch_loss_multiplier = tf.constant(patch_loss_multiplier, tf.float32)

    def get_patch(self):
        return self._denorm(self._patch, cast_uint=False).numpy()

    def first_pass(self, images):
        with tf.name_scope('first_pass'):
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
        with tf.name_scope('attack_pass'):
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
        with tf.name_scope('post_processing'):
            def single_batch_fn(element):
                return postprocess.nms(self.config, boxes[element], scores[element], classes[element], True)

            nms_boxes, nms_scores, nms_classes, nms_valid_len = tf.map_fn(single_batch_fn, tf.range(boxes.nrows()),
                                                                          fn_output_signature=(
                                                                          tf.TensorSpec((None, 4), dtype=tf.float32),
                                                                          tf.TensorSpec((None,), dtype=tf.float32),
                                                                          tf.TensorSpec((None,), dtype=tf.float32),
                                                                          tf.TensorSpec((), dtype=tf.int32)))
            nms_boxes = postprocess.clip_boxes(nms_boxes, self.config.image_size)
            nms_boxes = tf.RaggedTensor.from_tensor(nms_boxes, lengths=nms_valid_len)
            nms_scores = tf.RaggedTensor.from_tensor(nms_scores, lengths=nms_valid_len)
        return nms_boxes, nms_scores

    def call(self, images, *, training=True):
        boxes = self._labels = self.first_pass(images)

        with tf.GradientTape() as tape:
            self._images = self._patcher([boxes, images])
            boxes_pred, scores, classes = self.second_pass(self._images)
            loss = tf.reduce_sum(tf.reduce_max(scores, axis=1)) + self.tv_loss()

        self.add_loss(loss)

        boxes_pred, scores_pred = self._postprocessing(boxes_pred, scores, classes)
        self.add_metric(self._metric(boxes_pred, boxes), name='mean_asr')

        func = functools.partial(self.vis_images, boxes_pred, training)
        tf.cond(tf.equal(tf.math.floormod(self.cur_step, self.visualize_freq), tf.constant(0, tf.int64)),
                func, lambda: None)

        if training:
            return tape.gradient(loss, self._patch)

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
        images = self._denorm(images)
        tr = 'train' if training else 'val'
        with self.tb._writers[tr].as_default():
            if training:
                patch = self._denorm(self._patch)
                tf.summary.image('Current patch', patch[tf.newaxis], step=self.cur_step)
            tf.summary.image('Sample', images, step=self.cur_step, max_outputs=tf.shape(self._images)[0])

    def tv_loss(self):
        """TV loss"""
        strided = self._patch[:-1, :-1]
        return tf.reduce_sum(((strided - self._patch[:-1, 1:]) ** 2. +
                              (strided - self._patch[1:, :-1]) ** 2.) ** .5) * self._patch_loss_multiplier

    def train_step(self, inputs):
        self.cur_step = self.tb._train_step
        self.reset_state()
        grads = self(inputs)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        self.optimizer.apply_gradients([(grads, self._patch)])
        ret = {'loss': self.losses[0]}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def test_step(self, inputs):
        self.cur_step = self.tb._val_step
        self.reset_state()
        self(inputs, training=False)
        ret = {'loss': self.losses[0]}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def reset_state(self):
        self._images = self._labels = None

    def save_weights(self, filepath, **kwargs):
        tifffile.imwrite(filepath, self.get_patch())


class Denormalizer(tf.keras.layers.Layer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self.config = config

    def call(self, inputs, cast_uint=True):
        res = inputs * self.config.stddev_rgb + self.config.mean_rgb
        if cast_uint:
            return tf.cast(res, tf.uint8)
        return res


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch: tf.Variable, histmatcher, *args, aspect=1., origin=(.5, .5),
                 scale=.3, min_patch_height=60, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self._patch = patch
        self._matcher = histmatcher
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._boxes = None
        self.aspect = aspect
        self.origin = origin
        self.scale = scale
        self.min_patch_height = min_patch_height

    def add_patches_to_image(self, image):
        boxes = self._boxes[self._batch_counter]
        patch_boxes = tf.vectorized_map(functools.partial(self.create, image), boxes)
        patch_boxes = tf.gather_nd(patch_boxes, tf.where(tf.greater(patch_boxes[:, 2],
                                                                    tf.constant(self.min_patch_height, tf.float32))))
        # transform_decisions = tf.cast(tf.random.uniform(shape=(tf.shape(patch_boxes)[0], 3), minval=0, maxval=2,
        #                                                 dtype=tf.int32), tf.bool)

        self._patch_counter.assign(tf.constant(0))
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes)
        image, _ = tf.while_loop(lambda image, i: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]), loop_fn,
                                 [image, self._patch_counter])
        self._batch_counter.assign_add(tf.constant(1))
        return image

    # @tf.function
    def add_patch_to_image(self, patch_boxes, image, i):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        idx = tf.stack(tf.meshgrid(tf.range(ymin_patch, ymax), tf.range(xmin_patch, xmax), indexing='ij'), axis=-1)

        patch_bg = image[ymin_patch:ymax, xmin_patch:xmax]
        patch = self._matcher((self._patch, patch_bg))
        im = tf.image.resize(patch, tf.stack([patch_h, patch_w]))
        # im = tf.cond(transform_decisions[self._patch_counter, 0], lambda: tf.image.rot90(im), lambda: im)
        # im = tf.cond(transform_decisions[self._patch_counter, 1], lambda: tf.image.rot90(im), lambda: im)
        # im = tf.cond(transform_decisions[self._patch_counter, 2], lambda: tf.image.rot90(im), lambda: im)

        image = tf.tensor_scatter_nd_update(image, idx, im)
        self._patch_counter.assign_add(tf.constant(1))
        return [image, self._patch_counter]

    def create(self, image, bbox):
        ymin, xmin, ymax, xmax = tf.unstack(bbox, 4)

        h = ymax - ymin
        w = xmax - xmin

        patch_w = h * self.scale
        patch_h = self.aspect * patch_w

        orig_y = ymin + h / 2.
        orig_x = xmin + w / 2.

        ymin_patch = tf.maximum(orig_y - patch_h / 2., 0.)
        xmin_patch = tf.maximum(orig_x - patch_w / 2., 0.)

        shape = tf.cast(tf.shape(image), tf.float32)
        ymin_patch = tf.cond(tf.greater(ymin_patch + patch_h, shape[0]),
                             lambda: shape[0] - patch_h, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + patch_w, shape[1]),
                             lambda: shape[1] - patch_w, lambda: xmin_patch)

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w])

    def call(self, inputs, *args, **kwargs):
        self._boxes, images = inputs
        self._batch_counter.assign(tf.constant(0))
        result = tf.map_fn(self.add_patches_to_image, images)
        return result


def get_victim_model(download_model=False):
    if download_model:
        # Download checkpoint.
        util.download(MODEL)

    logger.info(f'Using model in {MODEL}')
    driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL)
    return driver.model


def main(download_model=False):
    from PIL import Image
    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    log_dir = util.ensure_empty_dir('log_dir')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    victim_model = get_victim_model(download_model)
    config_override = {'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}}
    model = PatchAttacker(victim_model, patch_loss_multiplier=0., config_override=config_override, visualize_freq=1)
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
    save_file = 'patch_{epoch:02d}_{val_mean_asr:.4f}.tiff'
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        steps_per_epoch=20,#train_len,
                        validation_steps=20,#val_len,
                        callbacks=[tb_callback,
                                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, save_file),
                                                                      monitor='val_mean_asr',
                                                                      verbose=1,
                                                                      save_best_only=False,
                                                                      save_weights_only=True,
                                                                      mode='auto',
                                                                      save_freq='epoch',
                                                                      options=None,
                                                                      initial_value_threshold=None,
                                                                      )
                                   ])
    patch = Image.fromarray(model.get_patch().astype('uint8'))
    patch.show('patch')


if __name__ == '__main__':
    main()
