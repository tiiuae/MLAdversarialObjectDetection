"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: patch generator
"""
import numpy as np
import tensorflow as tf


def define_generator():
    model = tf.keras.models.Sequential(name='generator')
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, input_dim=2))
    # model.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(512 * 512, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape((512, 512, 1)))
    model.compile(run_eagerly=False)

    return model


class ScaleGen(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(ScaleGen, self).__init__(*args, **kwargs, name='generator')
        self.l1 = tf.keras.layers.Dense(32)
        self.l2 = tf.keras.layers.BatchNormalization()
        self.l3 = tf.keras.activations.tanh
        self.l11 = tf.keras.layers.Dense(64)
        self.l12 = tf.keras.layers.BatchNormalization()
        self.l13 = tf.keras.activations.tanh
        self.op = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, dims, training=False):
        x = self.l1(dims)
        x = self.l2(x, training=training)
        x = self.l3(x)
        x = self.l11(x)
        x = self.l12(x, training=training)
        x = self.l13(x)
        return self.op(x)


def define_regressor():
    model = ScaleGen()
    model.build(input_shape=(None, 2))
    model.compile(run_eagerly=False)
    return model


def tv_loss(tensors):
    """TV loss"""
    strided = tensors[:, -1:, :-1]
    return tf.reduce_mean(((strided - tensors[:, -1:, 1:]) ** 2. +
                          (strided - tensors[:, 1:, :-1]) ** 2.) ** .5)


def centre_loss(delta):
    h, w, _ = tf.unstack(tf.cast(tf.shape(delta), tf.float32))
    indices = tf.cast(tf.where(tf.greater(delta, .5)), tf.float32)
    hind, wind, _ = tf.unstack(indices, axis=1)
    hind -= .5 * h
    wind -= .5 * w
    se = tf.math.square(hind) + tf.math.square(wind)
    return tf.reduce_max(se)


def main():
    model = define_generator()
    model.summary()
    print(model.predict(np.array([[-.1, -.2]])))
    print('=========')
    print(model.predict(np.array([[.2, -.1]])))


if __name__ == '__main__':
    main()
