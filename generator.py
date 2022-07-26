"""
©2022 Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: April 28, 2022

Purpose: contains attention U-net architecture implementation and adaptation for patch detection and recovery use-case
"""
from keras.backend import concatenate
from keras.layers import Layer, BatchNormalization, Activation, Dropout, Add, Multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

import utils


class UNetBackBone(Model):
    """Attention U-net backbone implementation container class"""

    def __init__(self, *args, n_filters=8, dropout=.2, batchnorm=True, **kwargs):
        """
        init
        :param args: superclass args
        :param n_filters: number of filters in the initial convolutional layer
        :param dropout: dropout value
        :param batchnorm: whether to use batch normalization in layers
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs)

        # encoder layers
        self.conv_blocks = [Conv2DBlock(name=f'conv{i}', n_filters=n_filters * (2 ** i), batchnorm=batchnorm,
                                        dropout=dropout)
                            for i in range(4)]
        self.conv_no_skip = Conv2DBlock(name='conv4', n_filters=n_filters * (2 ** 4), batchnorm=batchnorm,
                                        maxpool=False)

        # decoder layers
        self.deconv_blocks = []
        m = 8
        for i in range(4):
            self.deconv_blocks.append(Conv2DTransposeBlock(name=f'deconv{i}', n_filters=n_filters * m,
                                                           batchnorm=batchnorm, dropout=dropout))
            m /= 2

    def call(self, inputs, *args, training=False):
        """
        called during training and inference for each batch
        :param inputs: input images
        :param args: unused. kept for signature compatibility but not required
        :param training: boolean
        :return: output features
        """
        x = inputs
        encs = []

        # encoder path
        for layer in self.conv_blocks:
            enc, x = layer(x, training=training)
            encs.append(enc)
        x = self.conv_no_skip(x, training=training)

        encs = encs[::-1]

        # decoder path
        for enc, layer in zip(encs, self.deconv_blocks):
            x = layer([x, enc], training=training)
        return x


class PatchNeutralizer(UNetBackBone):
    """
    Generate targeted features for the patch detection use-case. The generated output is added to original images during
    attack_detection module training
    """

    def __init__(self, *args, **kwargs):
        """
        init
        :param args: superclass args
        :param kwargs: superclass kwargs
        """
        name = 'patch_neutralizer'
        super().__init__(*args, **kwargs, name=name)
        self.op = Conv2D(3, (1, 1), activation='tanh', name=f'{name}/output', kernel_initializer='he_normal')

    def call(self, inputs, *args, training=False):
        """
        called during training and inference for each batch
        :param inputs: input images
        :param args: unused. kept for signature compatibility but not required
        :param training: boolean
        :return: output features
        """
        x = super().call(inputs, training=training)
        return self.op(x)


class AttentionBlock(Layer):
    """
    implementation of convolutional attention for u-net decoder path. Please read the attention u-net paper for details:
    https://arxiv.org/pdf/1804.03999.pdf
    """

    def __init__(self, name, n_filters, **kwargs):
        """
        init
        :param name: name of the layer
        :param n_filters: number of features in the conv components
        :param kwargs: superclass kwargs
        """
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

    def call(self, input_tensor, *args, training=False, **kwargs):
        """
        called during training and inference for each batch
        :param input_tensor: incoming tensors
        :param args: unused. kept for signature compatibility but not required
        :param training: boolean
        :param kwargs: unused. kept for signature compatibility but not required
        :return: attention map features
        """
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
    """defines a single convolutional block in the encoder pathway of the u-net"""

    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, maxpool=True, **kwargs):
        """
        init
        :param args: supercalss args
        :param name: name of the conv block
        :param n_filters: number of features defined in this block
        :param kernel_size: conv kernel size
        :param dropout: dropout value
        :param batchnorm: whether to use batch normalization
        :param maxpool: whether to use max pooling
        :param kwargs: superclass kwargs
        """
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

    def call(self, input_tensor, *args, training=False):
        """
        called during training and inference for each batch
        :param input_tensor: input feature map
        :param args: unused. kept for signature compatibility but not required
        :param training: boolean
        :return: output feature map
        """
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
    """defines a single deconvolutional block in the decoder pathway of the u-net"""

    def __init__(self, *args, name, n_filters, kernel_size=3, dropout=None, batchnorm=True, attention=True, **kwargs):
        """
        init
        :param args: supercalss args
        :param name: name of the conv block
        :param n_filters: number of features defined in this block
        :param kernel_size: conv kernel size
        :param dropout: dropout value
        :param batchnorm: whether to use batch normalization
        :param attention: whether to use attention mechanism or if false define a plain u-net decoder
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs, name=name)
        self.l1 = Conv2DTranspose(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                  kernel_initializer='he_normal', padding='same', name=f'{name}/cnv')

        if attention:
            self.att = AttentionBlock(name=f'{name}/attention', n_filters=n_filters)
        self.attention = attention

        self.l2 = Dropout(dropout, name=f'{name}/do')
        self.l3 = Conv2DBlock(name=f'{name}/convblock', n_filters=n_filters, kernel_size=kernel_size, maxpool=False,
                              batchnorm=batchnorm)

    def call(self, input_tensor, *args, training=False):
        """
        called during training and inference for each batch
        :param input_tensor: input feature map
        :param args: unused. kept for signature compatibility but not required
        :param training: boolean
        :return: output feature map
        """
        fw, skip = input_tensor
        x = self.l1(fw)

        if self.attention:
            skip = self.att([x, skip], training=training)

        x = concatenate([x, skip])
        x = self.l2(x, training=training)
        x = self.l3(x)
        return x


def define_model(image_size: int, cls, *args, **kwargs):
    """
    factory method. defines a u-net model, builds and compiles
    :param image_size: image size in int
    :param cls: class to build
    :param args: passed to cls
    :param kwargs: passed to cls
    :return: model
    """
    output_shape = utils.parse_image_size(image_size)
    model = cls(*args, **kwargs)
    model.build(input_shape=(None, *output_shape, 3))
    model.compile(run_eagerly=False)
    return model


def test():
    """test only"""
    model = define_model(128, PatchNeutralizer)
    model.summary()


if __name__ == '__main__':
    test()
