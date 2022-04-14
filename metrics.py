"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 14, 2022

Purpose: Calculating recall after the attack
"""
import functools

import tensorflow as tf


class Recall:
    def __init__(self, iou_thresh=.5):
        self._boxes_gt = None
        self._iou_thresh = tf.constant(iou_thresh, tf.float32)
        self._result = None
        self._loop_var = tf.Variable(0, trainable=False, dtype=tf.int32)

    def __call__(self, boxes_pred, boxes_true):
        batch_size = tf.cast(boxes_true.nrows(), tf.int32)
        if self._result is None:
            self._result = tf.Variable(tf.zeros(shape=(batch_size,), dtype=tf.int32), trainable=False)
        else:
            self._result.assign(tf.zeros(shape=(batch_size,), dtype=tf.int32))

        self._boxes_gt = boxes_true
        gt_box_count = tf.vectorized_map(lambda boxes: tf.cast(boxes.nrows(), tf.int32), boxes_true)

        def map_fn(_):
            func = functools.partial(self._calc_valid, boxes_true[self._loop_var])
            self._result[self._loop_var].assign(tf.reduce_sum(tf.vectorized_map(func, boxes_pred[self._loop_var])))
            self._loop_var.assign_add(1)
            return [_]

        self._loop_var.assign(0)
        tf.while_loop(lambda _: tf.less(self._loop_var, batch_size), map_fn, [self._loop_var])

        return self._result / gt_box_count

    def _calc_valid(self, boxes_gt, box):

        def valid(box_gt):
            x_a = tf.math.maximum(box[0], box_gt[0])
            y_a = tf.math.maximum(box[1], box_gt[1])
            x_b = tf.math.minimum(box[2], box_gt[2])
            y_b = tf.math.minimum(box[3], box_gt[3])

            inter_area = tf.math.maximum(tf.constant(0.),
                                         x_b - x_a + tf.constant(1.)) * tf.math.maximum(tf.constant(0.),
                                                                                        y_b - y_a + tf.constant(1.))
            box_area = (box[2] - box[0] + tf.constant(1.)) * (box[3] - box[1] + tf.constant(1.))
            box_gt_Area = (box_gt[2] - box_gt[0] + tf.constant(1.)) * (box_gt[3] - box_gt[1] + tf.constant(1.))
            return tf.greater(inter_area / (box_area + box_gt_Area - inter_area), self._iou_thresh)

        return tf.cast(tf.reduce_any(tf.vectorized_map(valid, boxes_gt)), tf.int32)
