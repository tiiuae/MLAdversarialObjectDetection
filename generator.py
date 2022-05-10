"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: patch generator
"""
import numpy as np
from keras.models import Model
from keras.layers import Layer, BatchNormalization, Activation, Dropout, Reshape, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

import utils


class PatchGenerator(Model):
    def __init__(self, *args, n_filters=8, dropout=.05, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs, name='generator')
        self.conv_blocks = [Conv2DBlock(name=f'conv{i}', n_filters=n_filters * (2 ** i), batchnorm=batchnorm,
                                        dropout=dropout)
                            for i in range(6)]
        self.conv_blocks.append(Conv2DBlock(name='conv6', n_filters=n_filters * (2 ** 6), batchnorm=batchnorm,
                                            maxpool=False))
        self.fl = Flatten()
        self.dense = Dense(512, name='dense')
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = BatchNormalization(name=f'dense/bn')
        self.act = Activation('leaky_relu', name=f'dense/act')
        self.reshape = Reshape((1, 1, 512))
        self.deconv_blocks = []

        m = 32
        for i in range(9):
            self.deconv_blocks.append(Conv2DTransposeBlock(name=f'deconv{i}', n_filters=n_filters * m,
                                                           batchnorm=batchnorm, dropout=dropout))
            m /= 2

        self.op = Conv2D(1, (1, 1), activation='tanh', name='output')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_blocks:
            x = layer(x, training=training)
        x = self.fl(x)
        x = self.dense(x)
        if self.batchnorm:
            x = self.bn(x, training=training)
        x = self.act(x)
        x = self.reshape(x)
        for layer in self.deconv_blocks:
            x = layer(x, training=training)
        return self.op(x)


class Conv2DBlock(Layer):
    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, maxpool=True, **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                         padding='same', name=f'{name}/cnv1')
        if batchnorm:
            self.l2 = BatchNormalization(name=f'{name}/bn')
        self.batchnorm = batchnorm
        self.maxpool = maxpool
        self.dropout = dropout

        self.l3 = Activation('leaky_relu', name=f'{name}/act')
        if maxpool:
            self.l4 = MaxPooling2D((2, 2), name=f'{name}/mp')
        if dropout is not None:
            self.l5 = Dropout(dropout, name=f'{name}/do')

    def call(self, input_tensor, training=False):
        x = self.l1(input_tensor)
        if self.batchnorm:
            x = self.l2(x, training=training)
        x = self.l3(x)
        if self.maxpool:
            x = self.l4(x)
        if self.dropout:
            return self.l5(x, training=training)
        return x


class Conv2DTransposeBlock(Layer):
    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2DTranspose(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                  kernel_initializer='he_normal', padding='same', name=f'{name}/cnv')
        if batchnorm:
            self.l2 = BatchNormalization(name=f'{name}/bn')
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.l3 = Activation('leaky_relu', name=f'{name}/act')

        if dropout:
            self.l4 = Dropout(dropout, name=f'{name}/do')

    def call(self, input_tensor, training=False):
        x = self.l1(input_tensor)
        if self.batchnorm:
            x = self.l2(x, training=training)
        x = self.l3(x)
        if self.dropout:
            return self.l4(x, training=training)
        return x


class ScaleGen(Model):
    def __init__(self, *args, dropout=.05, **kwargs):
        super().__init__(*args, **kwargs, name='generator')
        self.l1 = Dense(128)
        self.l2 = BatchNormalization()
        self.l3 = Activation('leaky_relu')
        self.l4 = Dropout(rate=dropout)
        self.op = Dense(1, activation='sigmoid')

    def call(self, dims, training=False):
        x = self.l1(dims)
        x = self.l2(x, training=training)
        x = self.l3(x)
        x = self.l4(x, training=training)
        return self.op(x)


def define_generator(image_size):
    output_shape = utils.parse_image_size(image_size)
    model = PatchGenerator()
    model.build(input_shape=(None, *output_shape, 3))
    model.compile(run_eagerly=False)
    return model


def define_regressor():
    model = ScaleGen()
    model.build(input_shape=(None, 2))
    model.compile(run_eagerly=False)
    return model


def main():
    model = define_generator(480)
    model.summary()


if __name__ == '__main__':
    main()
