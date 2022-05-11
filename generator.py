"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: patch generator
"""
from keras.layers import Layer, BatchNormalization, Activation, Dropout, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model


class PatchGenerator(Model):
    def __init__(self, *args, n_filters=8, dropout=.05, batchnorm=True, **kwargs):
        super().__init__(*args, **kwargs, name='generator')
        self.deconv_blocks = []

        m = 32
        for i in range(9):
            self.deconv_blocks.append(Conv2DTransposeBlock(name=f'deconv{i}', n_filters=n_filters * m,
                                                           batchnorm=batchnorm, dropout=dropout))
            m /= 2

        self.op = Conv2D(1, (1, 1), activation='tanh', name='output')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.deconv_blocks:
            x = layer(x, training=training)
        return self.op(x)


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


def define_generator():
    model = PatchGenerator()
    model.build(input_shape=(None, 1, 1, 512))
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
