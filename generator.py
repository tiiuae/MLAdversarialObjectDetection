"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: patch generator
"""
from keras.backend import concatenate
from keras.layers import Layer, BatchNormalization, Activation, Dropout, Dense, Add, Multiply, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

import utils


class UNetBackBone(Model):
    def __init__(self, *args, n_filters=8, dropout=.2, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_blocks = [Conv2DBlock(name=f'conv{i}', n_filters=n_filters * (2 ** i), batchnorm=batchnorm,
                                        dropout=dropout)
                            for i in range(4)]
        self.conv_no_skip = Conv2DBlock(name='conv4', n_filters=n_filters * (2 ** 4), batchnorm=batchnorm,
                                        maxpool=False)

        self.deconv_blocks = []
        m = 8
        for i in range(4):
            self.deconv_blocks.append(Conv2DTransposeBlock(name=f'deconv{i}', n_filters=n_filters * m,
                                                           batchnorm=batchnorm, dropout=dropout))
            m /= 2

    def call(self, inputs, training=False):
        x = inputs
        encs = []
        for layer in self.conv_blocks:
            enc, x = layer(x, training=training)
            encs.append(enc)
        x = self.conv_no_skip(x, training=training)

        encs = encs[::-1]
        for enc, layer in zip(encs, self.deconv_blocks):
            x = layer([x, enc], training=training)
        return x


class NoiseGenerator(UNetBackBone):
    def __init__(self, *args, **kwargs):
        name = 'noise_generator'
        super().__init__(*args, **kwargs, name=name)
        self.op = Conv2D(3, (1, 1), activation='tanh', name=f'{name}/output', kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        x = super().call(inputs, training=training)
        return self.op(x)


class PatchGenerator(UNetBackBone):
    def __init__(self, *args, **kwargs):
        name = 'patch_generator'
        super().__init__(*args, **kwargs, name=name)
        self.op = Conv2D(3, (1, 1), activation='tanh', name=f'{name}/output', kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        x = super().call(inputs, training=training)
        patch = self.op(x)
        return patch


class AttentionBlock(Layer):
    def __init__(self, name, n_filters, **kwargs):
        super().__init__(**kwargs, name=name)
        self.l1 = Conv2D(n_filters, 1, padding='valid', name=f'{name}/cnv1')
        self.l2 = BatchNormalization(name=f'{name}/bn1')

        self.l3 = Conv2D(n_filters, 1, padding='valid', name=f'{name}/cnv2')
        self.l4 = BatchNormalization(name=f'{name}/bn2')

        self.l5 = Add(name=f'{name}/add')
        self.l6 = Activation('leaky_relu', name=f'{name}/act1')

        self.l7 = Conv2D(1, 1, padding='valid', name=f'{name}/conv3')
        self.l8 = BatchNormalization(name=f'{name}/bn3')
        self.l9 = Activation('sigmoid', name=f'{name}/act2')

        self.l10 = Multiply(name=f'{name}/mul')

    def __call__(self, input_tensor, *args, training=False, **kwargs):
        up_in, skip_in = input_tensor
        g = self.l1(up_in)
        g = self.l2(g, training=training)

        x = self.l3(skip_in)
        x = self.l4(x, training=training)

        x = self.l5([g, x])
        x = self.l6(x)

        x = self.l7(x)
        x = self.l8(x, training=training)
        x = self.l9(x)
        return self.l10([skip_in, x])


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
        return x


class Conv2DTransposeBlock(Layer):
    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, attention=True, **kwargs):
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2DTranspose(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                  kernel_initializer='he_normal', padding='same', name=f'{name}/cnv')

        if attention:
            self.att = AttentionBlock(name=f'{name}/attention', n_filters=n_filters)
        self.attention = attention

        self.l2 = Dropout(dropout, name=f'{name}/do')
        self.l3 = Conv2DBlock(name=f'{name}/convblock', n_filters=n_filters, kernel_size=kernel_size, maxpool=False,
                              batchnorm=batchnorm)

    def call(self, input_tensor, training=False):
        fw, skip = input_tensor
        x = self.l1(fw)

        if self.attention:
            skip = self.att([x, skip], training=training)

        x = concatenate([x, skip])
        x = self.l2(x, training=training)
        x = self.l3(x)
        return x


def define_generator(image_size, cls):
    output_shape = utils.parse_image_size(image_size)
    model = cls()
    model.build(input_shape=(None, *output_shape, 3))
    model.compile(run_eagerly=False)
    return model


def main():
    model = define_generator(128, PatchGenerator)
    model.summary()


if __name__ == '__main__':
    main()
