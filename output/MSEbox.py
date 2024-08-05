# Set headers and convert MSE to numeric values for plotting
import pandas as pd
from matplotlib import pyplot as plt

ae_mse_data = pd.read_csv('../result/PRJEB13870/MSE/ae_values.tsv', sep='\t', header=0)
gan_mse_data = pd.read_csv('../result/PRJEB13870/MSE/gan_values.tsv', sep='\t', header=0)
vae_mse_data = pd.read_csv('../result/PRJEB13870/MSE/vae_values.tsv', sep='\t', header=0)

# Convert MSE values from string to float
ae_mse_data['MSE'] = pd.to_numeric(ae_mse_data['MSE'])
gan_mse_data['MSE'] = pd.to_numeric(gan_mse_data['MSE'])
vae_mse_data['MSE'] = pd.to_numeric(vae_mse_data['MSE'])

# Create a boxplot to compare the MSE values of the three models
plt.figure(figsize=(10, 6))
plt.boxplot([ae_mse_data['MSE'], gan_mse_data['MSE'], vae_mse_data['MSE']], labels=['AE', 'VAE_GAN', 'VAE'])
plt.title('Comparison of MSE Across Models')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

