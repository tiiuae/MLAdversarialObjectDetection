"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 12, 2022

Purpose: data generator for training. does the preprocessing on CPU and does augmentations on the GPU on the fly. uses
tf.dataset API for parallel processing, prefetching and batching
"""
import functools
import math
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import hparams_config
import util
import utils

logger = util.get_logger(__name__)


class DataSequence(tf.keras.utils.Sequence):
    """Sequence subclass to define rescaling and preprocessing operations"""

    def __init__(self, img_dir, output_size, mean_rgb, stddev_rgb, *, file_list=None, shuffle=True):
        """
        init
        :param img_dir: image directory
        :param output_size: output image size in int
        :param mean_rgb: mean value for standardization
        :param stddev_rgb: std_dev value for standardization
        :param file_list: file_list used when not None and image_dir is ignored
        :param shuffle: boolean
        """
        super().__init__()
        self._img_dir = img_dir
        self._output_size = output_size
        self._mean_rgb = mean_rgb
        self._stddev_rgb = stddev_rgb
        self._flist = file_list or os.listdir(self._img_dir)
        self._shuffle = shuffle

    def _read_and_preprocess_file(self, filename):
        """
        read file and return preprocessed image array
        :param filename: filename
        :return: preprocessed image
        """
        im = _read_image(self._img_dir, filename)
        return self._map_fn(im)

    def _map_fn(self, image):
        """
        preprocessing function. rescale and pad image to size and standardize it to have values between -1 and 1
        :param image: image array
        :return: preprocessed image array
        """
        h, w, c = image.shape
        image = image.astype(float)
        image -= self._mean_rgb
        image /= self._stddev_rgb

        image_scale_y = self._output_size[0] / h
        image_scale_x = self._output_size[1] / w
        image_scale = min(image_scale_x, image_scale_y)
        scaled_height = int(h * image_scale)
        scaled_width = int(w * image_scale)

        scaled_image = cv2.resize(image, [scaled_width, scaled_height])
        output_image = np.zeros((*self._output_size, c))
        output_image[:scaled_height, :scaled_width, :] = scaled_image
        return output_image

    def __len__(self):
        """
        number of images or dataset size
        :return: dataset size
        """
        return len(self._flist)

    def __getitem__(self, idx):
        """
        access an item from the dataset in preprocessed form
        :param idx: index
        :return: preprocessed image array
        """
        flist = self._flist[idx]
        return self._read_and_preprocess_file(flist)

    def __call__(self):
        """
        makes this class a generator. yields a preprocessed image tensor from the dataset forever, randomly if
        self._shuffle is True else in order
        :yield: preprocessed image tensor
        """
        if self._shuffle:
            np.random.shuffle(self._flist)

        i = 0
        while True:
            image = self[i]
            yield tf.convert_to_tensor(image, dtype=tf.float32)
            i += 1
            if i == len(self):
                if self._shuffle:
                    np.random.shuffle(self._flist)
                i = 0


def _parse_line(line):
    """
    for reading txt labels one line at a time
    :param line: label line
    :return: array of bounding box and labels
    """
    return list(map(float, line.strip().split(' ')[1:]))


def _read_image(img_dir, filename):
    """
    read an image from img_dir
    :param img_dir: directory containing image
    :param filename: image filename
    :return: raw RGB image array
    """
    im = Image.open(os.path.join(img_dir, filename))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return np.asarray(im)


def filter_by_dims(img_dir, label_dir, max_area_ratio, filename):
    """
    filter a dataset by person bounding box sizes relative to the image size
    :param img_dir: image directory
    :param label_dir: label directory
    :param max_area_ratio: maximum area ratio of bounding box relative to image area. if the image contains even one
    bounding box above this ratio, the image will be dropped from the dataset
    :param filename: image filename
    :return: False if at least one bounding box is over the max_area_ratio
    """
    # return True
    im = _read_image(img_dir, filename)
    h, w, _ = im.shape
    filename = os.path.extsep.join([os.path.splitext(filename)[0], 'txt'])
    with open(os.path.join(label_dir, filename)) as f:
        for line in f.readlines():
            ymin, xmin, ymax, xmax = _parse_line(line)
            if ymin < 20 or xmin < 20 or ymax > h - 20 or xmax > w - 20:
                return False
            hp = ymax - ymin
            wp = xmax - xmin
            area_ratio = (hp * wp) / (h * w)
            if area_ratio >= max_area_ratio:
                return False
    return True


def partition(config, img_dir, label_dir, max_area_ratio=.1, train_split=0.9, *, batch_size=2, shuffle=True):
    """
    partition the dataset into train and validation set
    :param config: model configuration to read required image size from
    :param img_dir: image directory
    :param label_dir: label directory
    :param max_area_ratio: maximum area ratio of bounding box relative to image area to be passed to filtering function
    :param train_split: train set size relative to dataset size
    :param batch_size: batch size to use for creating tf.dataset optimizations
    :param shuffle: shuffle decision
    :return: dictionary containing train and val sets and their respective batch lengths (i.e., number of steps per
    epoch)
    """
    # get val set size
    val_split = 1. - train_split

    # filter dataset by size constraints
    logger.info('filtering dataset by label constraints...')
    func = functools.partial(filter_by_dims, img_dir, label_dir, max_area_ratio)
    file_list = list(filter(func, os.listdir(img_dir)))
    ds_size = len(file_list)
    logger.info(f'done. data size is {ds_size}')

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    logger.info(f'training on {train_size} images, validating on {val_size}')

    # define data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomContrast(.2)
    ])

    @tf.autograph.experimental.do_not_convert
    def get_tf_dataset(start, end, validation=False):
        """
        create tf.dataset pipeline adding processing optimizations and data augmentations
        :param start: start index in the dataset size
        :param end: end index in the dataset size
        :param validation: if creating validation set, do not attach augmentations to the data flow pipeline
        :return:
        """
        output_size = utils.parse_image_size(config.image_size)
        dseq = DataSequence(img_dir, output_size, config.mean_rgb, config.stddev_rgb,
                            file_list=file_list[start:end], shuffle=shuffle)
        ds = tf.data.Dataset.from_generator(dseq, output_signature=tf.TensorSpec(shape=(*output_size, 3),
                                                                                 dtype=tf.float32)
                                            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        if not validation:
            return ds.map(tf.image.random_flip_left_right). \
                map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE). \
                map(lambda im: tf.image.random_brightness(im, .2), num_parallel_calls=tf.data.AUTOTUNE). \
                map(lambda im: tf.clip_by_value(im, -1., 1.), num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    # crate datasets
    train_ds = get_tf_dataset(0, train_size)
    val_ds = get_tf_dataset(train_size, ds_size, validation=True)

    # return with batch lengths (number of steps per epoch)
    return {'train': {'dataset': train_ds, 'length': math.ceil(train_size / batch_size)},
            'val': {'dataset': val_ds, 'length': math.ceil(val_size / batch_size)}}


def test(download_model=False):
    """test only"""
    model_name = 'efficientdet-lite4'

    if download_model:
        # Download checkpoint.
        util.download(model_name)

    config = hparams_config.get_efficientdet_config(model_name)
    train_ds, val_ds, test_ds = partition(config, 'downloaded_images', 'labels')
    print([(x.shape, y.shape) for x, y in train_ds.take(2)])
    print('=============')
    print([(x.shape, y.shape) for x, y in val_ds.take(2)])
    print('=============')
    print([(x.shape, y.shape) for x, y in test_ds.take(2)])


if __name__ == '__main__':
    test()
