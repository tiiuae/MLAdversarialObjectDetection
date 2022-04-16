"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 14, 2022

Purpose: Calculating recall after the attack
"""
import functools

import tensorflow as tf


class AttackSuccessRate:
    def __init__(self, min_bbox_area, *, iou_thresh=.5, ):
        self._min_bbox_area = tf.constant(min_bbox_area, tf.float32)
        self._iou_thresh = tf.constant(iou_thresh, tf.float32)
        self._loop_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._result = None

    def __call__(self, boxes_pred, boxes_true):
        batch_size = tf.cast(boxes_true.nrows(), tf.int32)
        if self._result is None:
            self._result = tf.Variable(tf.zeros(shape=(batch_size,), dtype=tf.float32), trainable=False)
        else:
            self._result.assign(tf.zeros(shape=(batch_size,), dtype=tf.float32))

        def map_fn(_):
            boxes_gt = boxes_true[self._loop_var].to_tensor()
            boxes_gt_area = (boxes_gt[:, 2] - boxes_gt[:, 0]) * (boxes_gt[:, 3] - boxes_gt[:, 1])
            atk_attempts = tf.where(tf.greater(boxes_gt_area, self._min_bbox_area))
            boxes_gt = tf.gather_nd(boxes_gt, atk_attempts)
            boxes_gt_area = tf.gather_nd(boxes_gt_area, atk_attempts)

            atk_attempt_count = tf.cast(tf.shape(boxes_gt)[0], tf.float32) + tf.constant(1e-6)
            func = functools.partial(self._calc_valid, boxes_gt, boxes_gt_area)
            atk_failed_count = tf.reduce_sum(tf.map_fn(func, boxes_pred[self._loop_var])) + tf.constant(1e-6)
            atk_success_rate = tf.constant(1.) - atk_failed_count / atk_attempt_count

            tf.assert_equal(tf.greater_equal(atk_success_rate, tf.constant(0.)), tf.constant(True))

            self._result[self._loop_var].assign(atk_success_rate)
            self._loop_var.assign_add(1)
            return [_]

        self._loop_var.assign(0)
        tf.while_loop(lambda _: tf.less(self._loop_var, batch_size), map_fn, [self._loop_var])

        return tf.reduce_mean(self._result)

    def _is_valid(self, boxes_gt, boxes_gt_area, box, ind):
        box_gt = boxes_gt[ind]
        box_gt_area = boxes_gt_area[ind]
        x_a = tf.math.maximum(box[0], box_gt[0])
        y_a = tf.math.maximum(box[1], box_gt[1])
        x_b = tf.math.minimum(box[2], box_gt[2])
        y_b = tf.math.minimum(box[3], box_gt[3])

        inter_area = tf.math.maximum(tf.constant(0.),
                                     x_b - x_a + tf.constant(1.)) * tf.math.maximum(tf.constant(0.),
                                                                                    y_b - y_a + tf.constant(1.))
        box_area = (box[2] - box[0] + tf.constant(1.)) * (box[3] - box[1] + tf.constant(1.))
        return tf.greater(inter_area / (box_area + box_gt_area - inter_area), self._iou_thresh)

    def _calc_valid(self, boxes_gt, boxes_gt_area, box):
        fn = functools.partial(self._is_valid, boxes_gt, boxes_gt_area, box)
        return tf.cast(tf.reduce_any(tf.map_fn(fn, tf.range(tf.shape(boxes_gt)[0]), fn_output_signature=tf.bool)),
                       tf.float32)
