"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 14, 2022

Purpose: Calculating recall after the attack
"""
import functools

import tensorflow as tf


class AttackSuccessRate:
    def __init__(self, *, iou_thresh=.5, ):
        # self._min_bbox_height = tf.constant(min_bbox_height, tf.float32)
        self._iou_thresh = tf.constant(iou_thresh, tf.float32)
        self._loop_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._failed_count = tf.Variable(0., dtype=tf.float32, trainable=False)
        self._attempts_count = tf.Variable(0., dtype=tf.float32, trainable=False)

    def __call__(self, boxes_pred, boxes_true):
        batch_size = tf.cast(boxes_true.nrows(), tf.int32)
        self._failed_count.assign(tf.constant(0.))
        self._attempts_count.assign(tf.constant(0.))

        def map_fn(_):
            boxes_gt = boxes_true[self._loop_var]
            boxes_gt_height = boxes_gt[:, 2] - boxes_gt[:, 0]
            boxes_gt_width = boxes_gt[:, 3] - boxes_gt[:, 1]
            boxes_gt_area = boxes_gt_height * boxes_gt_width
            # atk_attempts = tf.where(tf.greater_equal(boxes_gt_height, self._min_bbox_height))
            # boxes_gt = tf.gather_nd(boxes_gt, atk_attempts)
            # boxes_gt_area = tf.gather_nd(boxes_gt_area, atk_attempts)

            atk_attempt_count = tf.cast(tf.shape(boxes_gt)[0], tf.float32) + tf.constant(1e-6)

            def atk_failed():
                return tf.reduce_sum(tf.cast(tf.reduce_any(tf.map_fn(func, boxes_pred[self._loop_var], dtype=tf.bool),
                                                           axis=0), tf.float32))

            func = functools.partial(self._calc_valid, boxes_gt, boxes_gt_area)
            atk_failed_count = tf.cond(tf.greater(tf.shape(boxes_pred[self._loop_var])[0], tf.constant(0)),
                                       atk_failed, lambda: tf.constant(0.)) + tf.constant(1e-6)

            self._failed_count.assign_add(atk_failed_count)
            self._attempts_count.assign_add(atk_attempt_count)
            self._loop_var.assign_add(1)
            return [_]

        self._loop_var.assign(0)
        tf.while_loop(lambda _: tf.less(self._loop_var, batch_size), map_fn, [self._loop_var])

        atk_success_rate = tf.constant(1.) - self._failed_count / self._attempts_count
        tf.assert_equal(tf.greater_equal(atk_success_rate, tf.constant(0.)), tf.constant(True))

        return atk_success_rate

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
        box_h = box[2] - box[0]
        box_w = box[3] - box[1]
        box_area = box_h * box_w
        # cond1 = tf.greater(box_h, self._min_bbox_height)
        cond2 = tf.greater(inter_area / (box_area + box_gt_area - inter_area), self._iou_thresh)
        # return tf.logical_and(cond1, cond2)
        return cond2

    def _calc_valid(self, boxes_gt, boxes_gt_area, box):
        fn = functools.partial(self._is_valid, boxes_gt, boxes_gt_area, box)
        return tf.map_fn(fn, tf.range(tf.shape(boxes_gt)[0]), fn_output_signature=tf.TensorSpec((), tf.bool))
