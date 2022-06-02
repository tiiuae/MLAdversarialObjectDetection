"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: patch generator
"""
from keras.backend import concatenate
from keras.layers import Layer, BatchNormalization, Activation, Dropout, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

import utils


class NoiseGenerator(Model):
    def __init__(self, *args, n_filters=8, dropout=.05, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs, name='generator')
        self.conv_blocks = [Conv2DBlock(name=f'conv{i}', n_filters=n_filters * (2 ** i), batchnorm=batchnorm,
                                        dropout=dropout)
                            for i in range(4)]
        self.conv_blocks.append(Conv2DBlock(name='conv4', n_filters=n_filters * (2 ** 4), batchnorm=batchnorm,
                                            maxpool=False))

        self.deconv_blocks = []
        m = 8
        for i in range(4):
            self.deconv_blocks.append(Conv2DTransposeBlock(name=f'deconv{i}', n_filters=n_filters * m,
                                                           batchnorm=batchnorm, dropout=dropout))
            m /= 2

        self.op = Conv2D(1, (1, 1), activation='tanh', name='output', kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        x = inputs
        encs = []
        for layer in self.conv_blocks:
            enc, x = layer(x, training=training)
            if enc is not None:
                encs.append(enc)

        encs = encs[::-1]
        for enc, layer in zip(encs, self.deconv_blocks):
            x = layer((x, enc), training=training)
        return self.op(x)


class Conv2DBlock(Layer):
    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, maxpool=True, **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                         padding='same', name=f'{name}/cnv1')
        if batchnorm:
            self.l2 = BatchNormalization(name=f'{name}/bn1')
        self.l3 = Activation('leaky_relu', name=f'{name}/act1')

        self.l4 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                         padding='same', name=f'{name}/cnv2')
        if batchnorm:
            self.l5 = BatchNormalization(name=f'{name}/bn2')
        self.l6 = Activation('leaky_relu', name=f'{name}/act2')

        self.batchnorm = batchnorm
        self.maxpool = maxpool
        self.dropout = dropout

        if maxpool:
            self.l7 = MaxPooling2D((2, 2), name=f'{name}/mp')
        if dropout is not None:
            self.l8 = Dropout(dropout, name=f'{name}/do')

    def call(self, input_tensor, training=False):
        x = self.l1(input_tensor)
        if self.batchnorm:
            x = self.l2(x, training=training)
        x = self.l3(x)
        x = self.l4(x)
        if self.batchnorm:
            x = self.l5(x, training=training)
        x = self.l6(x)
        if self.maxpool:
            f = self.l7(x)
            if not self.dropout:
                return x, f
            else:
                return x, self.l8(f, training=training)
        if self.dropout:
            return x, self.l8(x, training=training)
        return None, x


class Conv2DTransposeBlock(Layer):
    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2DTranspose(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                  kernel_initializer='he_normal', padding='same', name=f'{name}/cnv')
        self.l2 = Dropout(dropout, name=f'{name}/do')
        self.l3 = Conv2DBlock(name=f'{name}/convblock', n_filters=n_filters, kernel_size=kernel_size, maxpool=False,
                              batchnorm=batchnorm)

    def call(self, input_tensor, training=False):
        fw, skip = input_tensor
        x = self.l1(fw)
        x = concatenate([x, skip])
        x = self.l2(x, training=training)
        _, x = self.l3(x)
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
    model = NoiseGenerator()
    model.build(input_shape=(None, *output_shape, 3))
    return model


def define_regressor():
    model = ScaleGen()
    model.build(input_shape=(None, 1))
    model.compile(run_eagerly=False)
    return model


def main():
    model = define_generator(480)
    model.summary()


if __name__ == '__main__':
    main()
