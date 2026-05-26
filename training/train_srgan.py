import tensorflow as tf

from training.losses import generator_loss, discriminator_loss 
from training.metrics import psnr_metric, ssim_metric


@tf.function
def train_step_srgan(
    lr_images, #[0, 255]
    hr_images, #[0, 255]
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    content_loss_fn,
    lambda_adv=1e-3,
    lambda_pixel=0.0,
):
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

        sr_images = generator(lr_images, training=True) # takes inputs in [0, 255], produces outputs in [0, 255]

        #normalize to [0, 1] for discriminator
        sr_for_d = sr_images / 255.0
        hr_for_d = hr_images / 255.0

        #train generator
        d_real = discriminator(hr_for_d, training=True) # takes inputs in [0, 1], produces outputs in [0, 1]
        d_fake = discriminator(sr_for_d, training=True)

        g_total_loss, g_loss_dict = generator_loss(
            sr=sr_images,
            hr=hr_images,
            d_fake=d_fake,
            content_loss_fn=content_loss_fn,
            lambda_adv=lambda_adv,
            lambda_pixel=lambda_pixel,
        )

        #train discriminator
        d_fake_detached = discriminator(tf.stop_gradient(sr_for_d), training=True)
        d_loss = discriminator_loss(d_real, d_fake_detached)


    g_grads = g_tape.gradient(g_total_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

  
    psnr = psnr_metric(hr_for_d, sr_for_d) # takes inputs in [0, 1]
    ssim = ssim_metric(hr_for_d, sr_for_d) # takes inputs in [0, 1]

    return {
        "g_total": g_total_loss,
        "g_content": g_loss_dict["content"],
        "g_adv": g_loss_dict["adversarial"],
        "g_pixel": g_loss_dict["pixel"],
        "d_loss": d_loss,
        "psnr": psnr,
        "ssim": ssim,
    }
