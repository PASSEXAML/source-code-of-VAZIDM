import numpy as np
def train(self, data, epochs=1000, batch_size=32, validation_split=0.1):
    # 开始训练判别器和生成器
    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data.iloc[idx]
        z_samples = np.random.normal(0, 1, (batch_size, self.gan.latent_dim))
        fake_data = self.gan.generator.predict(z_samples)

        # 判别器输入真实数据和假数据
        d_loss_real = self.gan.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = self.gan.discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, self.gan.latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = self.gan.gan_model.train_on_batch(noise, valid_y)

        # 打印损失
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    # 准备输入和目标数据
    x = data  # 输入数据
    y_combined = data  # 用于重建的目标，即与输入数据相同
    y_discriminator = np.ones((data.shape[0], 1))  # 判别器目标，全为 1 表示希望生成数据被判别为真实

    # 使用 VAE-GAN 模型的联合损失来训练
    loss =self.model.fit(
        x,
        [y_combined, y_discriminator],  # 输出目标列表
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    return loss
