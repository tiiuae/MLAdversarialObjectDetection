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


def define_regressor():
    model = tf.keras.models.Sequential(name='generator')
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, input_dim=2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(run_eagerly=False)

    return model


def tv_loss(tensors):
    """TV loss"""
    strided = tensors[:, -1:, :-1]
    return tf.reduce_mean(((strided - tensors[:, -1:, 1:]) ** 2. +
                          (strided - tensors[:, 1:, :-1]) ** 2.) ** .5)


def main():
    model = define_generator()
    model.summary()
    print(model.predict(np.array([[-.1, -.2]])))
    print('=========')
    print(model.predict(np.array([[.2, -.1]])))


if __name__ == '__main__':
    main()
