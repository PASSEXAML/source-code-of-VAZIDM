import pandas as pd

# Load the newly uploaded microbiome_external.tsv file
original_data = pd.read_csv('../data/IBD_EXTERNEL/microbiome_external.tsv', sep='\t')

# Load the mean.tsv file again to ensure we have the latest data
ae_data = pd.read_csv('../data/IBD_EXTERNEL/mean.tsv', sep='\t')
denoised_data = pd.read_csv('../result/IBD_EXTERNEL/output_values.tsv', sep='\t')

# Extract the columns from microbiome_external_df that correspond to the columns in mean_df
# (excluding the first column 'clade_name' which is the same in both)
ae_sample_columns = ae_data.columns[1:]  # Exclude the first column 'clade_name'
extracted_columns_df = original_data[['# Feature / Sample'] + list(ae_sample_columns)]
ae_column_names = ['# Feature / Sample'] + [f"ae_{col}" for col in ae_data.columns[1:]]
ae_data.columns = ae_column_names
denoised_data.columns = ['# Feature / Sample'] + [f"denoised_{col}" for col in denoised_data.columns[1:]]

# First merge 'extracted_columns_df' and 'ae_data'
merged_df = pd.merge(extracted_columns_df, ae_data, on='# Feature / Sample', how='inner')

# Then merge the result with 'denoised_data'
merged_df = pd.merge(merged_df, denoised_data, on='# Feature / Sample', how='inner')


# Save the extracted data to a new file
merged_df.to_csv('../result/IBD_EXTERNEL/merged_data.tsv', index=False)
