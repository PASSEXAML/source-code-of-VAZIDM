import pandas as pd
import matplotlib.pyplot as plt

# 载入数据
ae_pca_path = '../result/IBD_PRISM/PCA/ae_pca_values.tsv'
gan_pca_path = '../result/IBD_PRISM/PCA/gan_pca_values.tsv'
vae_pca_path = '../result/IBD_PRISM/PCA/vae_pca_values.tsv'

ae_pca_data = pd.read_csv(ae_pca_path, sep='\t')
gan_pca_data = pd.read_csv(gan_pca_path, sep='\t')
vae_pca_data = pd.read_csv(vae_pca_path, sep='\t')

# 设置画图环境
plt.figure(figsize=(14, 8))

# 为每个数据集绘制散点图，使用不同颜色
plt.scatter(ae_pca_data['Principal Component 1'], ae_pca_data['Principal Component 2'], alpha=0.5, label='AE')
plt.scatter(gan_pca_data['Principal Component 1'], gan_pca_data['Principal Component 2'], alpha=0.5, label='VAE_GAN')
plt.scatter(vae_pca_data['Principal Component 1'], vae_pca_data['Principal Component 2'], alpha=0.5, label='VAE')

# 添加图表标签和标题
plt.title('PCA Comparison between AE, GAN, and VAE')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
