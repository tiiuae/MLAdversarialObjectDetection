"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: attack the person detector
"""
import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import train_data_generator
import util
import utils
from tf2 import postprocess, efficientdet_keras, infer_lib

logger = util.get_logger(__name__)
MODEL = 'efficientdet-lite4'


class PatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, *, visualize=False):
        super().__init__(name='Graph')
        self.model = model
        self.config = self.model.config
        self.model.config.override({'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}})
        patch_img = np.random.rand(256, 256, 3)
        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32,
                                  constraint=lambda x: tf.clip_by_value(x, 0., 1.))
        self._patcher = Patcher(self._patch, name='Patcher')
        self._grad_processor = GradientProcessor(self._patch, patch_loss_multiplier=.5, name='Gradient_Processor')
        self._images = None
        self._visualize = visualize

    def get_patch(self):
        return (self._patch.numpy() * 255.).astype('uint8')

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
        boxes, vis_scores, classes = postprocess.pre_nms(self.config.as_dict(), cls_outputs, box_outputs)
        person_indices = tf.equal(classes, tf.constant(0))  # taking postprocess.CLASS_OFFSET into account
        scores = tf.ragged.boolean_mask(vis_scores, person_indices)
        return scores

    # @tf.function
    def call(self, inputs, *, training=True):
        if isinstance(inputs, (tuple, list)):
            images, labels = inputs
        else:
            images, labels = inputs, None

        if labels is None:
            boxes = self.first_pass(images)
        else:
            boxes = labels

        if not training:
            if labels is not None:
                raise ValueError('Fatal: if labels are supplied then must be training')
            return boxes

        if self._images is None:
            self._images = tf.Variable(images, name='inp_image', dtype=tf.float32)
        else:
            self._images.assign(images)

        patch_boxes, transform_decisions = self._patcher([boxes, self._images])

        with tf.GradientTape() as tape:
            scores = self.second_pass(self._images)
            loss = tf.reduce_max(scores)

        gradients = tape.gradient(loss, self._images)
        gradients = self._grad_processor([gradients, patch_boxes, transform_decisions])
        return gradients, loss

    def train_step(self, image):
        grads, loss = self(image)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        self.optimizer.apply_gradients([(grads, self._patch)])
        return {'loss': loss}


class GradientProcessor(tf.keras.layers.Layer):
    def __init__(self, patch, *args, patch_loss_multiplier=1., **kwargs):
        super().__init__(*args, **kwargs)
        self._patch = patch
        self._batch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._patch_counter = tf.Variable(tf.constant(0), trainable=False)
        self._agg = tf.Variable(tf.zeros_like(patch), trainable=False)
        self._patch_boxes = None
        self._transform_decisions = None
        self._gradients = None
        self.patch_loss_multiplier = tf.constant(patch_loss_multiplier, tf.float32)

    def inner_loop(self, _):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(self._patch_boxes[self._batch_counter,
                                                                                        self._patch_counter], tf.int32),
                                                              4)
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        gradients = self._gradients[self._batch_counter, ymin_patch:ymax, xmin_patch:xmax]
        transform_decisions = self._transform_decisions[self._batch_counter, self._patch_counter]
        gradients = tf.cond(transform_decisions[2], lambda: tf.image.flip_up_down(gradients), lambda: gradients)
        gradients = tf.cond(transform_decisions[1], lambda: tf.image.flip_left_right(gradients), lambda: gradients)
        gradients = tf.cond(transform_decisions[0], lambda: tf.image.rot90(gradients, k=3), lambda: gradients)
        gradients = tf.image.resize(gradients, self._patch.shape[:-1])
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

    def patch_loss(self):
        """TV loss"""
        pixel_front = tf.concat([self._patch[:, 1:], self._patch[:, -1:]], axis=1)
        pixel_down = tf.concat([self._patch[1:, :], self._patch[-1:, :]], axis=0)
        return tf.reduce_sum(((self._patch - pixel_front) ** 2 +
                              (self._patch - pixel_down) ** 2) ** .5) * self.patch_loss_multiplier

    def call(self, inputs, *args, **kwargs):
        self._gradients, self._patch_boxes, self._transform_decisions = inputs
        self._batch_counter.assign(tf.constant(0))
        tf.while_loop(lambda _: tf.less(self._batch_counter, tf.cast(self._patch_boxes.nrows(), tf.int32)),
                      self.batch_loop, [self._batch_counter])
        with tf.GradientTape() as t:
            patch_loss = self.patch_loss()
        self._agg.assign_add(t.gradient(patch_loss, self._patch))
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
                                                       tf.constant(400))))
        transform_decisions = tf.greater(tf.random.uniform(shape=(tf.shape(patch_boxes)[0], 3)), tf.constant(.5))
        loop_fn = functools.partial(self.add_patch_to_image, patch_boxes)
        tf.while_loop(lambda _, i: tf.less(self._patch_counter, tf.shape(patch_boxes)[0]),
                      loop_fn, [self._patch_counter, transform_decisions])
        self._batch_counter.assign_add(tf.constant(1))
        return tf.RaggedTensor.from_tensor(tf.concat([patch_boxes, tf.cast(transform_decisions, tf.float32)], axis=1))

    def add_patch_to_image(self, patch_boxes, _, transform_decisions):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(patch_boxes[self._patch_counter], tf.int32))
        im = tf.image.resize(self._patch, tf.stack([patch_h, patch_w]))
        im = tf.cond(transform_decisions[self._patch_counter, 0], lambda: tf.image.rot90(im), lambda: im)
        im = tf.cond(transform_decisions[self._patch_counter, 1], lambda: tf.image.flip_left_right(im), lambda: im)
        im = tf.cond(transform_decisions[self._patch_counter, 2], lambda: tf.image.flip_up_down(im), lambda: im)

        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w
        self._images[self._batch_counter, ymin_patch:ymax, xmin_patch:xmax].assign(im)
        self._patch_counter.assign_add(tf.constant(1))
        return [self._patch_counter, transform_decisions]

    def create(self, bbox):
        ymin, xmin, h, w = tf.unstack(bbox, 4)

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
        result = tf.map_fn(self.add_patches_to_image, batch_boxes)
        patch_boxes = result[:, :, :4]
        transform_decisions = tf.cast(result[:, :, 4:], tf.bool)
        return patch_boxes, transform_decisions


def main(download_model=False):
    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if download_model:
        # Download checkpoint.
        util.download(MODEL)
        logger.info(f'Using model in {MODEL}')

    driver = infer_lib.KerasDriver(MODEL, debug=False, model_name=MODEL)
    model = PatchAttacker(driver.model)
    model.compile(optimizer='adam', run_eagerly=False)
    output_size = utils.parse_image_size(model.config.image_size)
    batch_size = 1
    data_gen = train_data_generator.COCOPersonsSequence('downloaded_images', 'labels', output_size,
                                                        model.config.mean_rgb, model.config.stddev_rgb,
                                                        batch_size=batch_size, shuffle=False)
    data_set = tf.data.Dataset.from_generator(data_gen, output_signature=(
        tf.TensorSpec(shape=(batch_size, *output_size, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(batch_size, None, 4), dtype=tf.float32)))
    history = model.fit(data_set, epochs=100)
    patch = model.get_patch()
    plt.imshow(patch)
    plt.show()


if __name__ == '__main__':
    main()
