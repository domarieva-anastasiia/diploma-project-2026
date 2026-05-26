import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


def build_vgg19(layer_name="block5_conv4"):
    """
    Builds a VGG19 model truncated at a given convolutional layer.
    The model is used only for feature extraction (frozen weights).
    """
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False

    outputs = vgg.get_layer(layer_name).output
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model


class ContentLoss(tf.keras.losses.Loss):
    """
    Perceptual (content) loss based on VGG19 feature maps.
    """
    def __init__(self, layer_name="block5_conv4", name="content_loss"):
        super().__init__(name=name)
        self.vgg = build_vgg19(layer_name)
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # VGG expects inputs in range [0, 255] and preprocessed
        # y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        # y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        y_true = preprocess_input(y_true * 255.0)
        y_pred = preprocess_input(y_pred * 255.0)

        true_features = self.vgg(y_true)
        pred_features = self.vgg(y_pred)

        return self.mse(true_features, pred_features)



def pixel_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(d_real, d_fake):
    """
    Discriminator loss for real and fake images.
    Works for both ImageGAN and PatchGAN outputs.
    """
    real_loss = bce(tf.ones_like(d_real), d_real)
    fake_loss = bce(tf.zeros_like(d_fake), d_fake)
    return real_loss + fake_loss


def generator_adversarial_loss(d_fake):
    """
    Generator adversarial loss: tries to fool the discriminator.
    """
    return bce(tf.ones_like(d_fake), d_fake)



def generator_loss(
    sr, #[0, 255]
    hr, #[0, 255]
    d_fake,
    content_loss_fn,
    lambda_adv=1e-3,
    lambda_pixel=0.0,
):
    """
    Full generator loss for SRGAN.

    Parameters:
    - sr: super-resolved image (G output)
    - hr: high-resolution ground truth
    - d_fake: discriminator prediction for sr
    - content_loss_fn: instance of ContentLoss
    - lambda_adv: weight for adversarial loss
    - lambda_pixel: weight for pixel loss (optional)
    """

    #normalize from [0, 255] to [0, 1] to content loss
    hr_norm = hr / 255.0
    sr_norm = sr / 255.0

    sr_norm_clip = tf.clip_by_value(sr_norm, 0.0, 1.0)
    hr_norm_clip = tf.clip_by_value(hr_norm, 0.0, 1.0)

    l_content = content_loss_fn(hr_norm_clip, sr_norm_clip)
    l_adv = generator_adversarial_loss(d_fake)

    total_loss = l_content + lambda_adv * l_adv

    if lambda_pixel > 0.0:
        l_pixel = pixel_loss(hr_norm_clip, sr_norm_clip)
        total_loss += lambda_pixel * l_pixel
    else:
        l_pixel = tf.constant(0.0)

    return total_loss, {
        "content": l_content,
        "adversarial": l_adv,
        "pixel": l_pixel,
    }
