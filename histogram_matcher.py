"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 19, 2022

Purpose: match histograms in tensorflow
"""
import functools

import tensorflow as tf


class HistogramMatcher(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        src, tgt = inputs
        h, w, _ = src.shape
        res = []
        for i in range(3):
            source, target = src[:, :, i:i+1], tgt[:, :, i:i+1]
            cdfsrc = self.equalize_histogram(source)
            cdftgt = self.equalize_histogram(target)

            fn = functools.partial(self.interpolate, cdftgt, tf.range(-1., 1., delta=1./256.))
            pxmap = tf.vectorized_map(fn, cdfsrc)

            fn = functools.partial(self.interpolate, cdfsrc, pxmap)
            pxmap = tf.vectorized_map(fn,  tf.reshape(source, (h*w)))
            res.append(tf.reshape(pxmap, (h, w)))
        return tf.stack(res, axis=2)

    @staticmethod
    def equalize_histogram(image):
        values_range = tf.constant([-1., 1.], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(image, values_range, 256)
        cdf = tf.cumsum(histogram)
        cdf_min = tf.reduce_min(cdf)

        img_shape = tf.shape(image)
        pix_cnt = img_shape[-3] * img_shape[-2]
        cdfimg = tf.cast(cdf - cdf_min, tf.float32) * 2. / tf.cast(pix_cnt - 1, tf.float32) - 1.
        return cdfimg

    @staticmethod
    def interpolate(dx_T, dy_T, x):
        with tf.name_scope('interpolate'):
            with tf.name_scope('neighbors'):
                delVals = dx_T - x
                ind_1 = tf.argmin(tf.abs(delVals))
                ind_0 = ind_1 - 1

            with tf.name_scope('calculation'):
                value = tf.cond(tf.less_equal(x, dx_T[0]),
                                lambda: dy_T[0],
                                lambda: tf.cond(
                                    tf.greater_equal(x, dx_T[-1]),
                                    lambda: dy_T[-1],
                                    lambda: (dy_T[ind_0] + (dy_T[ind_1] - dy_T[ind_0]) * (x - dx_T[ind_0]) /
                                             (dx_T[ind_1] - dx_T[ind_0]))
                                ))
        return value


def main():
    from PIL import Image
    import numpy as np
    burj1 = Image.open('burj_khalifa_day.jpg').convert('RGB').resize((640, 640))
    burj2 = Image.open('burj_khalifa_sunset.jpg').convert('RGB').resize((640, 640))
    burj1_tf = tf.convert_to_tensor(np.asarray(burj1), dtype=tf.float32)
    burj2_tf = tf.convert_to_tensor(np.asarray(burj2), dtype=tf.float32)

    burj1_tf -= 127.
    burj1_tf /= 127.
    burj2_tf -= 127.
    burj2_tf /= 127.

    res = HistogramMatcher().call((burj1_tf, burj2_tf))
    res = Image.fromarray((res.numpy() * 127. + 127.).astype('uint8'))

    burj1.show('source')
    burj2.show('reference')
    res.show('result')

    res1 = tf.convert_to_tensor(np.asarray(res))

    res1 -= 127.
    res1 /= 127.

    res1 = HistogramMatcher().call((res1, burj1_tf))
    res1 = Image.fromarray((res1.numpy() * 127. + 127.).astype('uint8'))
    res1.show('restored')


if __name__ == '__main__':
    main()
