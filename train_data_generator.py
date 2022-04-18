"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 12, 2022

Purpose: data generator for training physical adversarial attacker on COCO persons
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


class COCOPersonsSequence(tf.keras.utils.Sequence):

    def __init__(self, img_dir, label_dir, output_size, mean_rgb, stddev_rgb, *, file_list=None, shuffle=True):
        super().__init__()
        self._img_dir = img_dir
        self._label_dir = label_dir
        self._output_size = output_size
        self._mean_rgb = mean_rgb
        self._stddev_rgb = stddev_rgb
        self._flist = file_list or os.listdir(self._img_dir)
        self._shuffle = shuffle

    def _read_files(self, filename):
        boxes = []
        im = _read_image(self._img_dir, filename)
        filename = os.path.extsep.join([os.path.splitext(filename)[0], 'txt'])
        with open(os.path.join(self._label_dir, filename)) as f:
            for line in f.readlines():
                boxes.append(_parse_line(line))
        return self._map_fn(im, np.array(boxes))

    def clip_boxes(self, boxes):
        """Clip boxes to fit in an image."""
        ymin, xmin, ymax, xmax = np.hsplit(boxes, 4)
        ymin = np.clip(ymin, 0, self._output_size[0] - 1)
        xmin = np.clip(xmin, 0, self._output_size[1] - 1)
        ymax = np.clip(ymax, 0, self._output_size[0] - 1)
        xmax = np.clip(xmax, 0, self._output_size[1] - 1)
        boxes = np.hstack([ymin, xmin, ymax, xmax])
        return boxes

    def _map_fn(self, image, boxes):
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

        boxes *= image_scale
        boxes = self.clip_boxes(boxes)
        boxes = boxes[(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0.]  # sane boxes only allowed
        return output_image, boxes

    def __len__(self):
        return len(self._flist)

    def __getitem__(self, idx):
        flist = self._flist[idx]
        return self._read_files(flist)

    def __call__(self):
        if self._shuffle:
            np.random.shuffle(self._flist)

        i = 0
        while True:
            image, boxes = self[i]
            yield (tf.convert_to_tensor(image, dtype=tf.float32),
                   tf.RaggedTensor.from_tensor(tf.convert_to_tensor(boxes, dtype=tf.float32)))
            i += 1
            if i == len(self):
                if self._shuffle:
                    np.random.shuffle(self._flist)
                i = 0


def _parse_line(line):
    return list(map(float, line.strip().split(' ')[1:]))


def _read_image(img_dir, filename):
    im = Image.open(os.path.join(img_dir, filename))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return np.asarray(im)


def filter_by_dims(img_dir, label_dir, min_height_ratio, min_width_ratio, aspect, filename):
    return True
    im = _read_image(img_dir, filename)
    h, w, _ = im.shape
    filename = os.path.extsep.join([os.path.splitext(filename)[0], 'txt'])
    with open(os.path.join(label_dir, filename)) as f:
        for line in f.readlines():
            ymin, xmin, ymax, xmax = _parse_line(line)
            hp = ymax - ymin
            wp = xmax - xmin
            h_ratio = hp / h
            w_ratio = wp / w
            ratio = wp / (hp + 1e-12)  # to avoid zero div
            if h_ratio >= min_height_ratio and w_ratio >= min_width_ratio and ratio >= aspect:
                return True


def partition(config, img_dir, label_dir, min_height_ratio=.7, min_width_ratio=.1, aspect_ratio=.1,
              train_split=0.8, val_split=0.1, test_split=0.1, *, batch_size=2,
              shuffle=True):
    assert (train_split + test_split + val_split) == 1.

    logger.info('filtering dataset by label constraints...')
    func = functools.partial(filter_by_dims, img_dir, label_dir, min_height_ratio, min_width_ratio, aspect_ratio)
    file_list = list(filter(func, os.listdir(img_dir)))
    ds_size = len(file_list)
    logger.info(f'done. data size is {ds_size}')

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    logger.info(f'training on {train_size} images, validating on {val_size}')

    def get_tf_dataset(start, end):
        output_size = utils.parse_image_size(config.image_size)
        dseq = COCOPersonsSequence(img_dir, label_dir, output_size, config.mean_rgb, config.stddev_rgb,
                                   file_list=file_list[start:end], shuffle=shuffle)
        return tf.data.Dataset.from_generator(dseq, output_signature=(
            tf.TensorSpec(shape=(*output_size, 3), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32))).batch(batch_size).prefetch(10)

    train_ds = get_tf_dataset(0, train_size)
    val_ds = get_tf_dataset(train_size, train_size+val_size)
    test_ds = get_tf_dataset(train_size+val_size, ds_size)

    return {'train': {'dataset': train_ds, 'length': math.ceil(train_size / batch_size)},
            'val': {'dataset': val_ds, 'length': math.ceil(val_size / batch_size)},
            'test': {'dataset': test_ds, 'length': math.ceil((ds_size - val_size - train_size) / batch_size)}}


def main(download_model=False):
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
    main()
