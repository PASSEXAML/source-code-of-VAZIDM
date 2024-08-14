import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 加载数据
ae_df = pd.read_csv('../data/IBD_PRISM/mean.tsv', sep='\t')
original_df = pd.read_csv('../data/IBD_PRISM/microbiome_PRISM.tsv', sep='\t')
vae_df = pd.read_csv('../data/IBD_PRISM/output_values.tsv', sep='\t')
gan_df = pd.read_csv('../result/IBD_PRISM/output_values.tsv', sep='\t')

# 确保数据包含相同的样本进行比较
common_samples = list(
    set(ae_df.columns[1:]).intersection(set(original_df.columns[1:])).intersection(set(vae_df.columns[1:])))

# 过滤数据以仅包含这些共同的样本
mean_df_filtered = ae_df[['# Feature / Sample'] + common_samples]
original_df_filtered = original_df[['# Feature / Sample'] + common_samples]
vae_df_filtered = vae_df[['# Feature / Sample'] + common_samples]
without_df_filtered = vae_df[['# Feature / Sample'] + common_samples]


# 标准化数据
def normalize(df):
    normalized_df = df.copy()
    for column in df.columns[1:]:
        normalized_df[column] = df[column] / df[column].sum()
    return normalized_df


mean_normalized = normalize(mean_df_filtered)
original_normalized = normalize(original_df_filtered)
vae_normalized = normalize(vae_df_filtered)
without_normalized = normalize(without_df_filtered)

# 执行 Mann-Whitney U 测试
results_ae = []
results_vae = []
results_without = []

for clade in original_normalized['# Feature / Sample']:
    ae_values = mean_normalized[mean_normalized['# Feature / Sample'] == clade].iloc[:, 1:].values.flatten()
    original_values = original_normalized[original_normalized['# Feature / Sample'] == clade].iloc[:, 1:].values.flatten()
    vae_values = vae_normalized[vae_normalized['# Feature / Sample'] == clade].iloc[:, 1:].values.flatten()
    without_values = without_normalized[without_normalized['# Feature / Sample'] == clade].iloc[:, 1:].values.flatten()

    u_stat_ae, p_val_ae = mannwhitneyu(ae_values, original_values, alternative='two-sided')
    u_stat_vae, p_val_vae = mannwhitneyu(vae_values, original_values, alternative='two-sided')
    u_stat_without, p_val_without = mannwhitneyu(without_values, original_values, alternative='two-sided')

    results_ae.append({'# Feature / Sample': clade, 'u_stat': u_stat_ae, 'p_val': p_val_ae})
    results_vae.append({'# Feature / Sample': clade, 'u_stat': u_stat_vae, 'p_val': p_val_vae})
    results_without.append({'# Feature / Sample': clade, 'u_stat': u_stat_without, 'p_val': p_val_without})

# 将结果转换为DataFrame
results_ae_df = pd.DataFrame(results_ae)
results_vae_df = pd.DataFrame(results_vae)
results_without_df = pd.DataFrame(results_without)

# 准备数据
results_ae_df['-log10(p_val)'] = -np.log10(results_ae_df['p_val'])
results_vae_df['-log10(p_val)'] = -np.log10(results_vae_df['p_val'])
results_without_df['-log10(p_val)'] = -np.log10(results_without_df['p_val'])

# 绘制火山图
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# AE模型火山图
axes[0].scatter(results_ae_df['u_stat'], results_ae_df['-log10(p_val)'], color='pink', alpha=0.5)
axes[0].set_title('AE Model Volcano Plot')
axes[0].set_xlabel('U Statistic')
axes[0].set_ylabel('-log10(p-value)')

#VAE模型火山图
axes[1].scatter(results_without_df['u_stat'], results_without_df['-log10(p_val)'], color='orange', alpha=0.5)
axes[1].set_title('VAE Model Volcano Plot')
axes[1].set_xlabel('U Statistic')
axes[1].set_ylabel('-log10(p-value)')

# VAE-GAN模型火山图
axes[2].scatter(results_vae_df['u_stat'], results_vae_df['-log10(p_val)'], color='blue', alpha=0.5)
axes[2].set_title('VAZIDM Model Volcano Plot')
axes[2].set_xlabel('U Statistic')
axes[2].set_ylabel('-log10(p-value)')

plt.tight_layout()
plt.savefig('../result/IBD_PRISM/volcano_plot.png')
plt.show()
