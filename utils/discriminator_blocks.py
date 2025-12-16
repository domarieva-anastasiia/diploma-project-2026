from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU


def discriminator_block(
    x,
    filters,
    strides=1,
    batchnorm=True,
    momentum=0.8
):
    x = Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same"
    )(x)

    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)

    return LeakyReLU(alpha=0.2)(x)
