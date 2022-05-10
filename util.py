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
        draw_bounding_box_on_image_array(
            frame,
            ymin,
            xmin,
            ymax,
            xmax,
            color='green',
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
    strided = tensors[:, -1:, :-1]
    return tf.reduce_mean(((strided - tensors[:, -1:, 1:]) ** 2. +
                          (strided - tensors[:, 1:, :-1]) ** 2.) ** .5)
