import pandas as pd
from matplotlib import pyplot as plt


# Since the datasets appear to contain precomputed MSE values for samples, we can directly plot these.
# Load the data if not already loaded correctly
ae_data_path = '../result/PRJDB4871/MSE/ae_values.tsv'
gan_data_path = '../result/PRJDB4871/MSE/gan_values.tsv'
vae_data_path = '../result/PRJDB4871/MSE/vae_values.tsv'
ae_mse_data = pd.read_csv(ae_data_path, sep='\t')
gan_mse_data = pd.read_csv(gan_data_path, sep='\t')
vae_mse_data = pd.read_csv(vae_data_path, sep='\t')

# Assuming 'Sample' is the common identifier and 'MSE' contains the MSE values
# Merge these datasets for a comparative plot
mse_comparison = pd.merge(ae_mse_data, gan_mse_data, on='Sample', suffixes=('_AE', '_GAN'))
mse_comparison = pd.merge(mse_comparison, vae_mse_data, on='Sample')
mse_comparison.rename(columns={'MSE': 'MSE_VAE'}, inplace=True)

# Now plot the data
plt.figure(figsize=(12, 8))

plt.bar(mse_comparison['Sample'], mse_comparison['MSE_AE'], color='#f5a129', label='AE', alpha=0.6)
plt.bar(mse_comparison['Sample'], mse_comparison['MSE_VAE'], color='#82ef8b', label='VAE', alpha=0.6)
plt.bar(mse_comparison['Sample'], mse_comparison['MSE_GAN'], color='#82d4ef', label='VAZIDM', alpha=0.6)

plt.xlabel('Sample')
plt.ylabel('MSE')
plt.title('Comparison of MSE Across Different Models')
plt.xticks(rotation=90)
plt.legend()
plt.show()
