import pandas as pd

df = pd.read_csv('PRJDB4871/output_values.csv')

df.to_csv('PRJDB4871/output_values.tsv', sep='\t', index=False)