import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, PReLU


upsamples_per_scale = {
    2: 1,
    4: 2,
    8: 3
}

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def residual_block(block_input, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([block_input, x])
    return x


def build_srresnet(scale=4, num_filters=64, num_res_blocks=16):
    if scale not in upsamples_per_scale:
        raise ValueError(f"available scales are: {upsamples_per_scale.keys()}")

    num_upsamples = upsamples_per_scale[scale]

    lr = Input(shape=(None, None, 3)) # expects [0,1]

    x = Conv2D(num_filters, kernel_size=9, padding='same')(lr)
    x_1 = x
    x = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    #x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    #x = Conv2D(3, kernel_size=9, padding='same')(x)
    #sr = Lambda(denormalize_m11)(x)
    sr = Conv2D(3, kernel_size=9, padding="same", activation="sigmoid")(x)


    return Model(lr, sr)