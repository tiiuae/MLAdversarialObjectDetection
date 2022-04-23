"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 19, 2022

Purpose: match histograms in tensorflow
"""
import tensorflow as tf


class HistogramMatcher(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_val = None
        self._max_val = None

    def _rescale_0_1(self, img):
        return (img + 1.) * 127. / 255.

    def _rescale_back(self, img):
        return (img * 255. / 127.) - 1.

    def reset_state(self):
        self._min_val = self._max_val = None

    def call(self, inputs, **kwargs):
        self.reset_state()
        src, tgt = inputs
        src = self._rescale_0_1(src)
        tgt = self._rescale_0_1(tgt)
        src = tf.image.rgb_to_hsv(src)
        tgt = tf.image.rgb_to_hsv(tgt)
        h, w, _ = tf.unstack(tf.shape(src))
        floating_space = tf.clip_by_value(tf.range(0, 1.00001, delta=1. / 255.), 0., 1.)
        src_h, src_s, src_v = tf.unstack(src, axis=2)

        source, target = src_v, tgt[:, :, -1:]
        cdfsrc = self.equalize_histogram(source)
        cdftgt = self.equalize_histogram(target)
        pxmap = self.interpolate(cdftgt, floating_space, cdfsrc)
        pxmap = self.interpolate(floating_space, pxmap, tf.reshape(source, (h * w,)))
        pxmap = tf.reshape(pxmap, (h, w))
        return self._rescale_back(tf.image.hsv_to_rgb(tf.stack([src_h, src_s, pxmap], axis=2)))

    @staticmethod
    def equalize_histogram(image):
        values_range = tf.constant([0., 1.], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(image, values_range, 256)
        cdf = tf.cumsum(histogram)
        cdf_min = tf.reduce_min(cdf)

        img_shape = tf.shape(image)
        pix_cnt = img_shape[0] * img_shape[1]
        cdfimg = tf.cast(cdf - cdf_min, tf.float32) / tf.cast(pix_cnt - 1, tf.float32)
        return cdfimg

    @staticmethod
    def interpolate(dx_T, dy_T, x):
        with tf.name_scope('interpolate'):
            with tf.name_scope('neighbors'):
                delVals = dx_T - x[:, tf.newaxis]
                ind_1 = tf.argmax(tf.sign(delVals), axis=1)
                ind_0 = ind_1 - 1
                ind_0 = tf.clip_by_value(ind_0, 0, 255)

            with tf.name_scope('calculation'):
                values = tf.where(tf.less_equal(x, dx_T[0]),
                                  dy_T[0],
                                  tf.where(
                                      tf.greater_equal(x, dx_T[-1]),
                                      dy_T[-1],
                                      tf.gather(dy_T, ind_0) + (tf.gather(dy_T, ind_1) - tf.gather(dy_T, ind_0)) *
                                      (x - tf.gather(dx_T, ind_0)) / (tf.gather(dx_T, ind_1) - tf.gather(dx_T, ind_0)))
                                  )
                tf.debugging.assert_equal(tf.reduce_any(tf.logical_or(tf.math.is_inf(values),
                                                                      tf.math.is_nan(values))), False)
        return values


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

    res1 = tf.convert_to_tensor(np.asarray(res), dtype=tf.float32)

    res1 -= 127.
    res1 /= 127.

    res1 = HistogramMatcher().call((res1, burj1_tf))
    res1 = Image.fromarray((res1.numpy() * 127. + 127.).astype('uint8'))
    res1.show('restored')


if __name__ == '__main__':
    main()
