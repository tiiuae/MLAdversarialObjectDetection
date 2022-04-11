"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 05 April, 2022

Purpose: utiltity functions and classes
"""
import logging

import tensorflow as tf


def convert_to_hw_format(bbox):
    """
    :param bbox: bbox in ymin, xmin, ymax, xmax format
    :return: bbox in ymin, xmin, h, w format
    """
    ymin, xmin, ymax, xmax = bbox
    return ymin, xmin, ymax-ymin, xmax-xmin


def get_logger(name, level=logging.ERROR):
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
