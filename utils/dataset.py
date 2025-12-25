import tensorflow as tf
import tensorflow_datasets as tfds
from training.train_srgan import preprocess_srgan

def load_div2k(subset="train", batch_size=8, patch_size=96, scale=4):  #перша версія із нормалізацією
    """Завантажує DIV2K та повертає tf.data.Dataset з LR і HR патчами"""

    ds = tfds.load("div2k/bicubic_x4", split=subset, as_supervised=True)

    def preprocess(lr, hr):
        lr = tf.image.random_crop(lr, [patch_size, patch_size, 3])
        hr = tf.image.random_crop(hr, [patch_size*scale, patch_size*scale, 3])
        lr = tf.cast(lr, tf.float32) / 255.0
        hr = tf.cast(hr, tf.float32) / 255.0
        return lr, hr

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_div2k_srgan(subset="train", batch_size=4, hr_crop_size=256, scale=4):
    ds = tfds.load("div2k/bicubic_x4", split=subset, as_supervised=True)

    ds = ds.map(
        lambda lr, hr: preprocess_srgan(
            lr, hr,
            hr_crop_size=hr_crop_size,
            scale=scale
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
