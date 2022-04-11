"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 01, 2022

Purpose: attack the person detector
"""
import functools
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import adv_patch
import detector
import util
from tf2 import postprocess, efficientdet_keras

logger = util.get_logger(__name__)


class PatchAttacker(tf.keras.Model):
    """attack with malicious patches"""

    def __init__(self, model: efficientdet_keras.EfficientDetModel, *, visualize=False):
        super().__init__(name='Graph')
        self.model = model
        self.model.config.override({'nms_configs': {'iou_thresh': .5, 'score_thresh': .5}})
        patch_img = (np.random.rand(100, 100, 3) * 255.).astype('uint8')
        self._patch = tf.Variable(patch_img, trainable=True, name='patch', dtype=tf.float32)
        self._image = None
        self._visualize = visualize
        self._loop_var = tf.Variable(tf.constant(0), name='loop_var')

    # @tf.function
    def create(self, bbox, *, aspect=1., origin=(.5, .3), scale=.01):
        ymin, xmin, h, w = tf.unstack(bbox)

        patch_h = h * scale
        patch_w = aspect * patch_h

        ymin_patch = ymin + h * origin[1]
        xmin_patch = xmin + w * origin[0]

        ymin_patch = tf.cond(tf.greater(ymin_patch + patch_h, self._image.shape[1]),
                             lambda: self._image.shape[1] - patch_h, lambda: ymin_patch)
        xmin_patch = tf.cond(tf.greater(xmin_patch + patch_w, self._image.shape[2]),
                             lambda: self._image.shape[2] - patch_w, lambda: xmin_patch)

        # TODO: implement random rotation about BB center
        # TODO: BONUS - random rotation about image axis

        return tf.stack([ymin_patch, xmin_patch, patch_h, patch_w])

    def _loop_body(self, boxes, _):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(boxes[self._loop_var], tf.int32))
        im = tf.image.resize(self._patch, tf.stack([patch_h, patch_w]))

        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w

        self._loop_var.assign_add(tf.constant(1))
        self._image[0, ymin_patch:ymax, xmin_patch:xmax].assign(im)
        return [self._loop_var]

    def _back_loop(self, boxes, gradients, ta: tf.TensorArray):
        ymin_patch, xmin_patch, patch_h, patch_w = tf.unstack(tf.cast(boxes[self._loop_var], tf.int32))
        ymax = ymin_patch + patch_h
        xmax = xmin_patch + patch_w

        gradients = gradients[ymin_patch:ymax, xmin_patch:xmax]
        im = tf.image.resize(gradients, self._patch.shape[:-1])
        ta = ta.write(self._loop_var.value(), im)
        self._loop_var.assign_add(tf.constant(1))
        return [ta]

    def first_pass(self, images):
        # preprocess.
        # noinspection PyProtectedMember
        images, scales = self.model._preprocessing(images, self.model.config.image_size,
                                                   self.model.config.mean_rgb, self.model.config.stddev_rgb,
                                                   'infer')

        boxes, scores, classes, _ = self.model(images, pre_mode=None)
        person_indices = tf.where(tf.equal(classes, tf.constant(1.)))
        scores = tf.gather_nd(scores, person_indices)
        boxes = tf.gather_nd(boxes, person_indices)
        boxes_h = boxes[:, 0] - boxes[:, 2]
        boxes_w = boxes[:, 1] - boxes[:, 3]
        boxes_area = boxes_h * boxes_w
        valid_boxes = tf.where(tf.greater(boxes_area, tf.constant(300.)))
        boxes = tf.gather_nd(boxes, valid_boxes)
        scores = tf.gather_nd(scores, valid_boxes)
        if scales is not None:
            scales = tf.expand_dims(scales, -1)
            boxes = boxes * tf.cast(scales, boxes.dtype)
        return boxes, scores

    def second_pass(self, image):
        cls_outputs, box_outputs = self.model(image, pre_mode=None, post_mode=None)
        cls_outputs = postprocess.to_list(cls_outputs)
        box_outputs = postprocess.to_list(box_outputs)
        boxes, scores, classes = postprocess.pre_nms(self.model.config.as_dict(), cls_outputs, box_outputs)
        person_indices = tf.where(tf.equal(classes, tf.constant(0)))  # taking postprocess.CLASS_OFFSET into account
        person_scores = tf.gather_nd(scores, person_indices)
        if self._visualize:
            boxes = tf.gather_nd(boxes, person_indices)
            classes = tf.gather_nd(classes, person_indices)

            def single_batch_fn(element):
                return postprocess.nms(self.model.config, element[0], element[1], element[2], True)

            nms_boxes, nms_scores, nms_classes, nms_valid_len = postprocess.batch_map_fn(
                single_batch_fn, [boxes, scores, classes])
            nms_boxes = postprocess.clip_boxes(nms_boxes, self.model.config['image_size'])
            return nms_boxes, nms_scores, nms_classes, nms_valid_len

        return person_scores

    # @tf.function
    def call(self, image, *, training=True):
        boxes, scores = self.first_pass(image)

        if not training:
            return boxes, scores

        image = tf.cast(image, tf.float32)

        if self._image is None:
            self._image = tf.Variable(image, name='inp_image', dtype=tf.float32)
        else:
            self._image.assign(image)

        patch_boxes = tf.map_fn(self.create, boxes)
        self._loop_var.assign(tf.constant(0))
        loop_fn = functools.partial(self._loop_body, patch_boxes)
        tf.while_loop(lambda _: tf.less(self._loop_var, tf.shape(patch_boxes)[0]), loop_fn, [self._loop_var])

        # noinspection PyProtectedMember
        image, scales = self.model._preprocessing(self._image, self.model.config.image_size,
                                                  self.model.config.mean_rgb, self.model.config.stddev_rgb,
                                                  'infer')

        with tf.GradientTape() as tape:
            tape.watch(image)
            outs = self.second_pass(image)
            if self._visualize:
                boxes, scores, classes, _ = outs
            else:
                scores = outs
            loss = tf.reduce_max(scores)

        gradients = tf.squeeze(tape.gradient(loss, image))

        ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, name='ta')
        self._loop_var.assign(tf.constant(0))
        if scales is not None:
            scales = tf.expand_dims(scales, -1)
            patch_boxes = patch_boxes * tf.cast(tf.constant(1.) / scales, patch_boxes.dtype)
        loop_fn = functools.partial(self._back_loop, patch_boxes, gradients)
        ta, = tf.while_loop(lambda _: tf.less(self._loop_var, tf.shape(patch_boxes)[0]), loop_fn, [ta])
        gradients = tf.reduce_sum(ta.stack(), axis=0)

        if self._visualize:
            if scales is not None:
                scales = tf.expand_dims(scales, -1)
                boxes = boxes * tf.cast(scales, boxes.dtype)
            return gradients, loss, boxes, scores
        return gradients, loss

    def train_step(self, image):
        outs = self(image)
        if self._visualize:
            grads, loss, boxes, scores = outs
        else:
            grads, loss = outs
        self.optimizer.apply_gradients([(grads, self._patch)])

        if self._visualize:
            logger.debug(f'loss: {loss}')
            return boxes, scores

        return {'loss': loss}


def main():
    det = detector.Detector(download_model=False)
    atk = PatchAttacker(det.driver.model, visualize=True)
    atk.compile(optimizer='adam')

    # noinspection PyShadowingNames
    logger = util.get_logger(__name__, logging.DEBUG)

    from streaming import Stream
    import matplotlib.pyplot as plt
    stream = Stream()
    for image in stream.playing():
        # atk.fit(np.expand_dims(image, 0), epochs=100, batch_size=1)
        logger.debug('read frame')
        boxes, scores = atk.train_step(tf.convert_to_tensor(np.expand_dims(image, 0)))
        cv2.imshow('Frame', atk._patch.numpy().astype(np.uint8))
        # # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
