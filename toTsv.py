import pandas as pd

df = pd.read_csv('./result/new_PRJEB13870/output_values.csv')

df.to_csv('./result/new_PRJEB13870/output_values.tsv', sep='\t', index=False)