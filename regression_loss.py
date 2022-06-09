"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 6, 2022

Purpose: calculating bounding box regression without priors
"""
import functools

import tensorflow as tf

import util


class DIOULoss:
    def __init__(self):
        self._loop_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._failed_count = tf.Variable(0., dtype=tf.float32, trainable=False)

    def __call__(self, boxes_pred, boxes_true):
        batch_size = tf.cast(boxes_true.nrows(), tf.int32)
        self._failed_count.assign(tf.constant(0.))

        def map_fn(_):
            boxes_gt = boxes_true[self._loop_var]
            boxes_gt_height = boxes_gt[:, 2] - boxes_gt[:, 0]
            boxes_gt_width = boxes_gt[:, 3] - boxes_gt[:, 1]
            boxes_gt_area = boxes_gt_height * boxes_gt_width

            def atk_failed():
                dious = tf.reduce_min(tf.map_fn(func, boxes_pred[self._loop_var]), axis=0)
                return tf.reduce_sum(dious)

            func = functools.partial(self._calc_valid, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width)
            atk_failed_count = tf.cond(tf.greater(tf.shape(boxes_pred[self._loop_var])[0], tf.constant(0)),
                                       atk_failed, lambda: tf.constant(0.)) + tf.constant(1e-6)

            self._failed_count.assign_add(atk_failed_count)
            self._loop_var.assign_add(1)
            return [_]

        self._loop_var.assign(0)
        tf.while_loop(lambda _: tf.less(self._loop_var, batch_size), map_fn, [self._loop_var])

        return self._failed_count

    def _is_valid(self, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width, box, ind):
        box_gt = boxes_gt[ind]
        box_gt_area, box_gt_height, box_gt_width = boxes_gt_area[ind], boxes_gt_height[ind], boxes_gt_width[ind]
        return util.diou_loss(box_gt, box_gt_area, box_gt_height, box_gt_width, box)

    def _calc_valid(self, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width, box):
        fn = functools.partial(self._is_valid, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width, box)
        return tf.map_fn(fn, tf.range(tf.shape(boxes_gt)[0]), dtype=tf.float32)
