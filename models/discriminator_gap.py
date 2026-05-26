import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Lambda, Reshape
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
    x = Reshape((1, 1, 512))(x)
    x = Conv2D(1, kernel_size=1)(x)
    x = Reshape((1,))(x)
    x = tf.keras.activations.sigmoid(x)

    return Model(x_in, x, name="Discriminator_GAP")
