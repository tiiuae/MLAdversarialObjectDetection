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

import requests
import tensorflow as tf


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
