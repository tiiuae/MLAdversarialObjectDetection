"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 05 April, 2022

Purpose: utiltity functions
"""
import logging
import os
import shutil
import tarfile

import cv2
import requests
import tensorflow as tf


def allow_direct_imports_from(dirname):
    """
    allows to write "import xyz" code without worrying about the directory structure. this function must be called to
    include the directory containing xyz in system path before writing "import xyz"
    :param dirname: dirname to include import from
    """
    import sys
    if dirname not in sys.path:
        sys.path.append(dirname)


allow_direct_imports_from('automl/efficientdet')
from automl.efficientdet.tf2 import infer_lib
from visualize.vis_utils import draw_bounding_box_on_image_array


@tf.function
def sign(tensor):
    """
    reimplementation of tf.sign operation due to a tensorflow bug that causes inconsistent behaviour on eager and
    non-eager execution modes
    :param tensor: tensor
    :return: equivalent of tf.sign(tensor) with consistent behaviour on eager and non-eager execution
    """
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
    """
    download an object detection model wights from the web
    :param m: model name
    """
    if m not in os.listdir():
        fname = f'{m}.tgz' if m.find('lite') else f'{m}.tar.gz'
        r = requests.get(f'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/{fname}')
        with open(fname, 'wb') as f:
            f.write(r.content)
        with tarfile.open(fname) as f:
            f.extractall()


def ensure_empty_dir(dirname):
    """
    create a directory if it doesn't exist or if it does clear all files within it
    :param dirname: directory name to process
    :return: reflects back directory name (not needed)
    """
    try:
        os.makedirs(dirname)
    except FileExistsError:
        shutil.rmtree(dirname, ignore_errors=True)
        os.makedirs(dirname)
    return dirname


def draw_boxes(frame, bb, sc):
    """
    draw bounding boxes and scores on top of image array in different box colors representing score value
    red box for low score
    yellow box for medium score
    green box for high score
    :param frame: frame
    :param bb: boxes list
    :param sc: scores list
    :return: decorated frame
    """
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


def puttext(img, text, pos, **txt_kwargs):
    """
    put text on image array with an off-white text shadow so that it remains visible regardless of the image background
    :param img: image array
    :param text: text string
    :param pos: psotion of text (x, y)
    :param txt_kwargs: cv2.put_text kwargs
    :return:
    """
    font_scale = txt_kwargs.get('font_scale')
    font_color = (150, 150, 150)
    thickness = txt_kwargs.get('thickness') + 1
    line_type = txt_kwargs.get('line_type')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text,
                pos,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)
    font_color = txt_kwargs.get('font_color')
    thickness = txt_kwargs.get('thickness')
    cv2.putText(img, text,
                pos,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)


def filter_by_thresh(bb, sc, score_thresh):
    """
    filter scores by threshold and then select remaining bounding boxes and scores
    :param bb: bounding box list
    :param sc: scores list
    :param score_thresh: threshold score to filter against
    :return: selected bounding boxes and scores
    """
    inds = [s >= score_thresh for s in sc]
    sc = [sc[i] for i in range(len(sc)) if inds[i]]
    bb = [bb[i] for i in range(len(bb)) if inds[i]]
    return bb, sc


def get_victim_model(model, download_model=False):
    """
    get keras model to be attacked
    :param model: model name
    :param download_model: whether to download model weights from the web, if false it assumes that the model weights
    are already downloaded and present as filename {model}.tgz
    :return: keras model
    """
    if download_model:
        # Download checkpoint.
        download(model)
    driver = infer_lib.KerasDriver(model, debug=False, model_name=model)
    return driver.model


def self_weightd_binary_ce(y_true, y_pred):
    """
    binary cross entropy weighted by number of examples in each class
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: self weighted binary ce loss
    """
    false_targets = tf.where(tf.not_equal(y_true, 0.), 1., 0.)

    # calculate weight factor based on class representation
    alpha_factor = 1. - tf.reduce_mean(false_targets)

    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    p_t = tf.where(tf.equal(y_true, 1.), y_pred, 1. - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1.), alpha_factor, 1. - alpha_factor)
    cross_entropy = -tf.math.log(p_t)

    loss = alpha_t * cross_entropy
    loss = tf.reduce_sum(tf.reduce_mean(loss, axis=1))
    return loss
