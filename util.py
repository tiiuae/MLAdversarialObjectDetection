"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 05 April, 2022

Purpose: utiltity functions and classes
"""
import logging
import os
import shutil
import tarfile

import cv2
import requests
import shapely.geometry
import tensorflow as tf

from tf2 import infer_lib
from visualize.vis_utils import draw_bounding_box_on_image_array


@tf.function
def sign(tensor):
    """replacement of tf.sign due to bug with inconsistent behaviour on eager and non-eager execution"""
    with tf.name_scope('sign'):
        gt = tf.where(tf.greater(tensor, 0.), 1., 0.)
        lt = tf.where(tf.less(tensor, 0.), -1., 0.)
    return gt + lt


def convert_to_hw_format(bbox):
    """
    :param bbox: bbox in ymin, xmin, ymax, xmax format
    :return: bbox in ymin, xmin, h, w format
    """
    ymin, xmin, ymax, xmax = bbox
    return ymin, xmin, ymax-ymin, xmax-xmin


def get_logger(name, level=logging.INFO):
    """
    get a logger by name or create one if it doesn't exist
    :param name: the name
    :param level: logging level
    :return: logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger  # it already exists!

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def download(m):
    if m not in os.listdir():
        fname = f'{m}.tgz' if m.find('lite') else f'{m}.tar.gz'
        r = requests.get(f'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/{fname}')
        with open(fname, 'wb') as f:
            f.write(r.content)
        with tarfile.open(fname) as f:
            f.extractall()


def ensure_empty_dir(dirname):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        shutil.rmtree(dirname, ignore_errors=True)
        os.makedirs(dirname)
    return dirname


def draw_boxes(frame, bb, sc):
    for box, score in zip(bb, sc):
        ymin, xmin, ymax, xmax = box
        color = 'green' if score >= .75 else 'yellow' if score >= .65 else 'red'
        draw_bounding_box_on_image_array(
            frame,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=1,
            display_str_list=[f'person: {int(100 * score)}%'], use_normalized_coordinates=False)

    return frame


def convert_to_shapely_format(box):
    ymin, xmin, ymax, xmax = box
    return [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]


def calculate_iou(box_1, box_2):
    poly_1 = shapely.geometry.Polygon(convert_to_shapely_format(box_1))
    poly_2 = shapely.geometry.Polygon(convert_to_shapely_format(box_2))
    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area


def puttext(img, text, pos, **txt_kwargs):
        font_scale = txt_kwargs.get('font_scale')
        font_color = txt_kwargs.get('font_color')
        thickness = txt_kwargs.get('thickness')
        line_type = txt_kwargs.get('line_type')
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = pos
        cv2.putText(img, text,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    line_type)


def filter_by_thresh(bb, sc, score_thresh):
    inds = [s >= score_thresh for s in sc]
    sc = [sc[i] for i in range(len(sc)) if inds[i]]
    bb = [bb[i] for i in range(len(bb)) if inds[i]]
    return bb, sc


def centre_loss(delta):
    h, w, _ = tf.unstack(tf.cast(tf.shape(delta), tf.float32))
    indices = tf.cast(tf.where(tf.greater(delta, .5)), tf.float32)
    hind, wind, _ = tf.unstack(indices, axis=1)
    hind -= .5 * h
    wind -= .5 * w
    se = tf.math.square(hind) + tf.math.square(wind)
    return tf.reduce_max(se)


def tv_loss(tensors):
    """TV loss"""
    strided = tensors[-1:, :-1]
    return tf.reduce_mean(((strided - tensors[-1:, 1:]) ** 2. +
                          (strided - tensors[1:, :-1]) ** 2.) ** .5)

@tf.function
def cmyk_to_rgb(patch):
    c, m, y, k = tf.unstack(patch, axis=2)
    key = (1. - k)
    r = 255. * (1. - c) * key
    g = 255. * (1. - m) * key
    b = 255. * (1. - y) * key
    return tf.stack([r, g, b], axis=2)


def get_victim_model(model, download_model=False):
    if download_model:
        # Download checkpoint.
        download(model)
    driver = infer_lib.KerasDriver(model, debug=False, model_name=model)
    return driver.model


def diou_loss(b1, b1_area, b1_height, b1_width, b2):
        zero = 0.
        # shape = tf.maximum(b1.bounding_shape(), b2.bounding_shape())
        # b1 = b1.to_tensor()
        # b2 = b2.to_tensor()
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        # b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        # b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        # b1_area = b1_width * b1_height
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
        # enclose_area = enclose_width * enclose_height
        # giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return 1. - diou


def binary_ce(y_true, y_pred):
    mask_targets = tf.where(tf.not_equal(y_true, 0.), 1., 0.)
    alpha_factor = 1. - tf.reduce_mean(mask_targets)

    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    p_t = tf.where(tf.equal(y_true, 1.), y_pred, 1. - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1.), alpha_factor, 1. - alpha_factor)
    cross_entropy = -tf.math.log(p_t)

    loss = alpha_t * cross_entropy
    loss = tf.reduce_sum(tf.reduce_mean(loss, axis=1))
    return loss
