"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: November 17, 2022

Purpose: create tfrecords for the visdrone dataset
"""
import tensorflow as tf

import automl.efficientdet.dataset.tfrecord_util as tfrecord_util


def main():
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    tfrecord_util.int64_feature(height),
                'image/width':
                    tfrecord_util.int64_feature(width),
                'image/filename':
                    tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                    tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                    tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                    tfrecord_util.bytes_feature(encoded_jpg),
                'image/format': tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                    tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                    tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                    tfrecord_util.float_list_feature(ymax),
                'image/object/area':
                    tfrecord_util.float_list_feature(area),
                'image/object/class/text':
                    tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                    tfrecord_util.int64_list_feature(classes),
                'image/object/difficult':
                    tfrecord_util.int64_list_feature(difficult_obj),
                'image/object/truncated':
                    tfrecord_util.int64_list_feature(truncated),
                'image/object/view':
                    tfrecord_util.bytes_list_feature(poses),
            }))
    return example


if __name__ == '__main__':
    main()
