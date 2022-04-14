"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 12, 2022

Purpose: data generator for training physical adversarial attacker on COCO persons
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import hparams_config
import util
import utils


class COCOPersonsSequence(tf.keras.utils.Sequence):

    def __init__(self, img_dir, label_dir, output_size, mean_rgb, stddev_rgb, *, shuffle=True):
        super().__init__()
        self._img_dir = img_dir
        self._label_dir = label_dir
        self._output_size = output_size
        self._mean_rgb = mean_rgb
        self._stddev_rgb = stddev_rgb
        self._flist = os.listdir(self._img_dir)
        self._shuffle = shuffle

    @staticmethod
    def _parse_line(line):
        return list(map(float, line.strip().split(' ')[1:]))

    def _read_image(self, filename):
        im = Image.open(os.path.join(self._img_dir, filename))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        return np.asarray(im)

    def _read_files(self, filename):
        boxes = []
        im = self._read_image(filename)
        filename = os.path.extsep.join([os.path.splitext(filename)[0], 'txt'])
        with open(os.path.join(self._label_dir, filename)) as f:
            for line in f.readlines():
                boxes.append(self._parse_line(line))
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

        for i in range(len(self)):
            image, boxes = self[i]
            yield (tf.convert_to_tensor(image, dtype=tf.float32),
                   tf.RaggedTensor.from_tensor(tf.convert_to_tensor(boxes, dtype=tf.float32)))


def partition(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1.

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def main(download_model=False):
    model_name = 'efficientdet-lite4'

    if download_model:
        # Download checkpoint.
        util.download(model_name)

    config = hparams_config.get_efficientdet_config(model_name)
    output_size = utils.parse_image_size(config.image_size)
    batch_size = 2
    dseq = COCOPersonsSequence('downloaded_images', 'labels', output_size, config.mean_rgb, config.stddev_rgb)
    dgen = tf.data.Dataset.from_generator(dseq, output_signature=(
        tf.TensorSpec(shape=(*output_size, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32))).batch(batch_size).prefetch(2)
    print([(x.shape, y.shape) for x, y in dgen.take(2)])


if __name__ == '__main__':
    main()
