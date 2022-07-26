"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: June 6, 2022

Purpose: calculating bounding box regression without priors for use in attack
Current status: unused. didn't get improvement in results or attack success rate. Module kept for future reuse
TODO: This code may not be optimal. can be optimized. too many gpu loops here. need to find a more efficient way
"""
import functools

import keras.backend
import tensorflow as tf


class InverseDIOULoss:
    """
    Define inverse distance based IOU loss function as inverse box regression objective to keep the bounding boxes
    away from target
    """

    def __init__(self):
        """init"""
        self._loop_var = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._idiou = tf.Variable(0., dtype=tf.float32, trainable=False)

    def __call__(self, boxes_pred, boxes_true):
        """
        makes this class callable. calculate InvDIOU loss
        :param boxes_pred: predicted bounding boxes
        :param boxes_true: target bounding boxes
        :return: InvDIOU for the batch
        """
        batch_size = tf.cast(boxes_true.nrows(), tf.int32)
        self._idiou.assign(tf.constant(0.))

        # gpu loop
        def map_fn(_):
            """
            for each image in batch
            :param _: loop var
            :return: InvDIOU for image
            """
            boxes_gt = boxes_true[self._loop_var]
            boxes_gt_height = boxes_gt[:, 2] - boxes_gt[:, 0]
            boxes_gt_width = boxes_gt[:, 3] - boxes_gt[:, 1]
            boxes_gt_area = boxes_gt_height * boxes_gt_width

            def check_box_pred():
                """
                process predicted bounding boxes and target bounding boxes for this image
                :return: InvDIOU for the image
                """
                dious = tf.reduce_max(tf.map_fn(func, boxes_pred[self._loop_var]), axis=0)
                return tf.reduce_sum(dious)

            func = functools.partial(self._calc_pred_box, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width)
            image_diou = tf.cond(tf.greater(tf.shape(boxes_pred[self._loop_var])[0], tf.constant(0)),
                                 check_box_pred, lambda: tf.constant(0.)) + keras.backend.epsilon()

            self._idiou.assign_add(image_diou)
            self._loop_var.assign_add(1)
            return [_]

        self._loop_var.assign(0)
        tf.while_loop(lambda _: tf.less(self._loop_var, batch_size), map_fn, [self._loop_var])

        return self._idiou

    # gpu loop
    def _calc_pred_box_to_gt_box(self, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width, box, ind):
        """
        for each ground truth box per predicted box
        :param boxes_gt: ground truth boxes for this image
        :param boxes_gt_area: ground truth boxes area for this image
        :param boxes_gt_height: ground truth boxes height for this image
        :param boxes_gt_width:  ground truth boxes width for this image
        :param box: one predicted bounding box
        :param ind: ground truth bounding box index
        :return: InvDIOU loss for this box against a ground truth box represented by boxes_gt[ind]
        """
        box_gt = boxes_gt[ind]
        box_gt_area, box_gt_height, box_gt_width = boxes_gt_area[ind], boxes_gt_height[ind], boxes_gt_width[ind]
        return 1. - self.diou_loss(box_gt, box_gt_area, box_gt_height, box_gt_width, box)

    # gpu loop
    def _calc_pred_box(self, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width, box):
        """
        for each predicted box in image
        :param boxes_gt: ground truth boxes for this image
        :param boxes_gt_area: ground truth boxes area for this image
        :param boxes_gt_height: ground truth boxes height for this image
        :param boxes_gt_width:  ground truth boxes width for this image
        :param box: one predicted bounding box
        :return: a tf tensor (array) of InvDIOU losses for this box against all ground truth boxes
        """
        fn = functools.partial(self._calc_pred_box_to_gt_box, boxes_gt, boxes_gt_area, boxes_gt_height, boxes_gt_width,
                               box)
        return tf.map_fn(fn, tf.range(tf.shape(boxes_gt)[0]), dtype=tf.float32)

    @staticmethod
    def diou_loss(b1, b1_area, b1_height, b1_width, b2):
        """
        DIOU loss implementation. please read DIOU loss paper for details: https://arxiv.org/pdf/1911.08287.pdf
        :param b1: bounding box 1
        :param b1_area: bounding box 1 area
        :param b1_height: bounding box 1 height
        :param b1_width: bounding box 1 width
        :param b2: bounding box 2
        :return: distance IOU loss between b1 and b2
        """
        zero = 0.
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)

        b1_centre_xy = tf.stack([b1_ymin + b1_height, b1_xmin + b1_width], axis=-1)
        b2_centre_xy = tf.stack([b2_ymin + b2_height, b2_xmin + b2_width], axis=-1)
        center_dist = tf.reduce_sum((b1_centre_xy - b2_centre_xy) ** 2., axis=-1)

        enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
        enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
        enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
        enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_diag = tf.reduce_sum(tf.stack([enclose_height, enclose_width], axis=-1) ** 2., axis=-1)
        diou = iou - tf.math.divide_no_nan(center_dist, enclose_diag)
        return 1. - diou
