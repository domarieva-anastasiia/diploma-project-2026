import tensorflow as tf

from training.losses import generator_loss, discriminator_loss 
from training.metrics import psnr_metric, ssim_metric


@tf.function
def train_step_srgan(
    lr_images,
    hr_images,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    content_loss_fn,
    lambda_adv=1e-3,
    lambda_pixel=0.0,
):
    
    with tf.GradientTape(persistent=True) as tape:
        # ----------------------------------
        # Generator forward
        # ----------------------------------
        sr_images = generator(lr_images, training=True)

        # ----------------------------------
        # Discriminator forward
        # ----------------------------------
        d_real = discriminator(hr_images, training=True)
        d_fake = discriminator(sr_images, training=True)

        # ----------------------------------
        # Losses
        # ----------------------------------
        d_loss = discriminator_loss(d_real, d_fake)

        g_total_loss, g_loss_dict = generator_loss(
            sr=sr_images,
            hr=hr_images,
            d_fake=d_fake,
            content_loss_fn=content_loss_fn,
            lambda_adv=lambda_adv,
            lambda_pixel=lambda_pixel,
        )

    # ----------------------------------
    # Gradients & updates
    # ----------------------------------
    g_grads = tape.gradient(g_total_loss, generator.trainable_variables)
    d_grads = tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # ----------------------------------
    # Metrics
    # ----------------------------------
    psnr = psnr_metric(hr_images, sr_images)
    ssim = ssim_metric(hr_images, sr_images)

    return {
        "g_total": g_total_loss,
        "g_content": g_loss_dict["content"],
        "g_adv": g_loss_dict["adversarial"],
        "g_pixel": g_loss_dict["pixel"],
        "d_loss": d_loss,
        "psnr": psnr,
        "ssim": ssim,
    }
