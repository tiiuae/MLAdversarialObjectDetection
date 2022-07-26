"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 19, 2022

Purpose: match scene brightness of patch with target image. This is done by converting RGB to YUV colormap and then
performing intensity matching on Y channel before converting back to RGB. Done on GPU
"""
import tensorflow as tf

import util


class BrightnessMatcher(tf.keras.layers.Layer):
    """match scene brightness to patch"""

    def __init__(self, *args, **kwargs):
        """
        init
        :param args: super args
        :param kwargs: super kwargs
        """
        super().__init__(*args, trainable=False, **kwargs)

    @staticmethod
    def _rescale_0_1(img):
        """
        rescale image between 0 and 1 from input image which is between -1 and 1
        :param img: input image with intensity between -1 and 1
        :return: rescaled image with intensity between 0 and 1
        """
        return (img + tf.constant(1.)) * tf.constant(127. / 255.)

    @staticmethod
    def _rescale_back(img):
        """
        rescale image between -1 and 1 from input image which is between 0 and 1
        :param img: input image with intensity between 0 and 1
        :return: rescaled image with intensity between -1 and 1
        """
        return img * tf.constant(255. / 127.) - tf.constant(1.)

    @tf.function
    def call(self, inputs, **kwargs):
        """
        called by outside class that wants to use brightness matching functionality
        :param inputs: input image
        :param kwargs: unused. kept for signature consistency
        :return: intensity matched image
        """
        src, tgt = inputs

        # rescale
        src = self._rescale_0_1(src)
        tgt = self._rescale_0_1(tgt)

        # convert to YUV colormap
        src = tf.image.rgb_to_yuv(src)
        tgt = tf.image.rgb_to_yuv(tgt)

        # align distribution means on Y channel
        source, target = src[:, :, 0], tgt[:, :, 0]
        source_mean = tf.reduce_mean(source)
        target_mean = tf.reduce_mean(target)
        pxmap = tf.clip_by_value(source - source_mean + target_mean, 0., 1.)

        # recombine and convert to RGB
        res = [pxmap, src[..., 1], src[..., 2]]
        res = tf.clip_by_value(tf.image.yuv_to_rgb(tf.stack(res, axis=2)), 0., 1.)

        # rescale back
        res = self._rescale_back(res)
        return res


class HistogramMatcher(BrightnessMatcher):
    """
    perform a more thorough histogram matching on the Y channel so the distribution of Y channel is aligned instead
    of just the mean as is done by its superclass, the brightness matcher class
    """

    @tf.function
    def call(self, inputs, **kwargs):
        """
        called by outside class that wants to use brightness matching functionality
        :param inputs: input image
        :param kwargs: unused. kept for signature consistency
        :return: intensity matched image
        """
        src, tgt = inputs

        # rescale
        src = self._rescale_0_1(src)
        tgt = self._rescale_0_1(tgt)

        # convert to YUV
        src = tf.image.rgb_to_yuv(src)
        tgt = tf.image.rgb_to_yuv(tgt)

        h, w, _ = tf.unstack(tf.shape(src))
        floating_space = tf.clip_by_value(tf.range(0., 1.00001, delta=1. / 255., dtype=tf.float32), 0., 1.)

        # perform histogram specification on Y channel
        source, target = src[:, :, 0], tgt[:, :, 0]
        cdfsrc = self.equalize_histogram(source)
        cdftgt = self.equalize_histogram(target)
        pxmap = self.interpolate(cdftgt, floating_space, cdfsrc)
        pxmap = self.interpolate(floating_space, pxmap, tf.reshape(source, (h * w,)))
        pxmap = tf.reshape(pxmap, (h, w))

        # recombine and convert to RGB
        res = [pxmap, src[..., 1], src[..., 2]]
        res = tf.clip_by_value(tf.image.yuv_to_rgb(tf.stack(res, axis=2)), 0., 1.)
        res = self._rescale_back(res)
        return res

    @staticmethod
    def equalize_histogram(image):
        """
        equalize Y channel on the YUV image. part of histogram specification process
        :param image: input image with Y channel only
        :return: equalized Y channel
        """
        values_range = tf.constant([0., 1.], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(image, values_range, tf.constant(256))
        cdf = tf.cumsum(histogram)
        cdf_min = tf.reduce_min(cdf)

        img_shape = tf.shape(image)
        pix_cnt = img_shape[0] * img_shape[1]
        cdfimg = tf.cast(cdf - cdf_min, tf.float32) / tf.cast(pix_cnt - tf.constant(1), tf.float32)
        return cdfimg

    @staticmethod
    def interpolate(dx_T, dy_T, x):
        """
        histogram specification require interpolation of intensity values. here I have manually implemented for tf
        as tf does not have an inbuilt library function for histogram matching
        :param dx_T: source cdf
        :param dy_T: target cdf
        :param x: interpolation points
        :return: interpolated pixel intensity map
        """
        with tf.name_scope('interpolate'):
            with tf.name_scope('neighbors'):
                delVals = dx_T - x[:, tf.newaxis]
                ind_1 = tf.argmax(util.sign(delVals), axis=1)
                ind_0 = ind_1 - tf.constant(1, tf.int64)
                ind_0 = tf.clip_by_value(ind_0, tf.constant(0, tf.int64), tf.constant(255, tf.int64))

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


def test():
    """test only"""
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

    res = HistogramMatcher()((burj1_tf, burj2_tf))
    res = Image.fromarray((res.numpy() * 127. + 127.).astype('uint8'))

    burj1.show('source')
    burj2.show('reference')
    res.show('result')

    res1 = tf.convert_to_tensor(np.asarray(res), dtype=tf.float32)

    res1 -= 127.
    res1 /= 127.

    res1 = HistogramMatcher()((res1, burj1_tf))
    res1 = Image.fromarray((res1.numpy() * 127. + 127.).astype('uint8'))
    res1.show('restored')


if __name__ == '__main__':
    test()
