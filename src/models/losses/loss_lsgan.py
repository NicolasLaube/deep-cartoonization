def lsgan_loss(
    discriminator, real, fake, scale=1, channel=32, patch=False, name="discriminator"
):

    real_logit = discriminator(
        real, scale, channel, name=name, patch=patch, reuse=False
    )
    fake_logit = discriminator(fake, scale, channel, name=name, patch=patch, reuse=True)

    g_loss = tf.reduce_mean((fake_logit - 1) ** 2)
    d_loss = 0.5 * (
        tf.reduce_mean((real_logit - 1) ** 2) + tf.reduce_mean(fake_logit ** 2)
    )

    return d_loss, g_loss
