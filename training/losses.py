import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real, fake):
    return bce(tf.ones_like(real), real) + bce(tf.zeros_like(fake), fake)

def generator_adversarial_loss(fake):
    return bce(tf.ones_like(fake), fake)
