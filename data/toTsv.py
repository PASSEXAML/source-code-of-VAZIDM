import pandas as pd

df = pd.read_csv('microbiome_external.csv')

df.to_csv('microbiome_external.tsv', sep='\t', index=False)