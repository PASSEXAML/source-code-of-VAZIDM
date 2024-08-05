import pandas as pd
from sklearn.metrics import r2_score

# Load the updated original data
updated_original_data_path = '../data/IBD_PRISM/microbiome_PRISM.tsv'
updated_original_data = pd.read_csv(updated_original_data_path, sep='\t')

# Load the updated predicted data
updated_predicted_data_path = '../result/IBD_PRISM/output_values.tsv'
updated_predicted_data = pd.read_csv(updated_predicted_data_path, sep='\t')

# Identify common sample columns between the updated original and predicted datasets
updated_common_samples = updated_original_data.columns.intersection(updated_predicted_data.columns)

# Filter data to only include common samples for comparison
updated_original_data_common = updated_original_data[updated_common_samples]
updated_predicted_data_common = updated_predicted_data[updated_common_samples]

# Calculate R-squared values for each common sample
updated_r_squared_values = {}
for sample in updated_original_data_common.columns[1:]:  # Exclude the '# Feature / Sample' column
    r_squared = r2_score(updated_original_data_common[sample], updated_predicted_data_common[sample])
    updated_r_squared_values[sample] = r_squared

# Convert updated R-squared values to a DataFrame for better visualization
updated_r_squared_df = pd.DataFrame(list(updated_r_squared_values.items()), columns=['Sample', 'R-squared'])

# Save the updated R-squared values
updated_r_squared_df.to_csv('../result/IBD_PRISM/R_2/gan_r_squared_values.tsv', sep='\t', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Create a bar plot for R-squared values
plt.figure(figsize=(14, 8))
r_squared_plot = sns.barplot(x='Sample', y='R-squared', data=updated_r_squared_df)
plt.xticks(rotation=90)  # Rotate the sample labels for better visibility
plt.title('R-squared Values for Each Sample')
plt.xlabel('Sample ID')
plt.ylabel('R-squared Value')
plt.show()




