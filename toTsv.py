import pandas as pd

df = pd.read_csv('./result/IBD_PRISM/output_values.csv')

df.to_csv('./result/IBD_PRISM/output_values.tsv', sep='\t', index=False)