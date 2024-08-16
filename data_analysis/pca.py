import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = '../data/PRJEB40200/mean.tsv'
data = pd.read_csv(file_path, sep='\t')

# Transpose the dataframe to have features as columns
data_t = data.set_index('clade_name').transpose()

# Standardizing the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_t)

# Applying PCA
pca = PCA(n_components=2)  # we choose 2 components for visualization purposes
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df.to_csv('../result/PRJEB40200/PCA/ae_pca_values.tsv', sep='\t', index=False)
# Plot the first two principal components
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of the Dataset')
plt.grid(True)
plt.show()
