import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

from utils.discriminator_blocks import discriminator_block



def build_discriminator_gap(hr_crop_size):

    x_in = Input(shape=(hr_crop_size, hr_crop_size, 3))

    x = discriminator_block(x_in, 64, batchnorm=False)
    x = discriminator_block(x, 64, strides=2)

    x = discriminator_block(x, 128)
    x = discriminator_block(x, 128, strides=2)

    x = discriminator_block(x, 256)
    x = discriminator_block(x, 256, strides=2)

    x = discriminator_block(x, 512)
    x = discriminator_block(x, 512, strides=2)

    x = GlobalAveragePooling2D()(x)
    x = Conv2D(1, kernel_size=1)(tf.expand_dims(tf.expand_dims(x, 1), 1))
    x = tf.squeeze(x, axis=[1, 2])
    x = tf.keras.activations.sigmoid(x)

    return Model(x_in, x, name="Discriminator_GAP")
