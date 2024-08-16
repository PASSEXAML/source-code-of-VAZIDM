import pandas as pd
# 删除全为0的行
micro_data = pd.read_csv("./PRJNA375935/PRJNA375935.tsv", sep='\t', index_col=0)
micro_data = micro_data.loc[~(micro_data==0).all(axis=1)]
micro_data = micro_data.loc[~(micro_data==1.0).all(axis=1)]
micro_data.to_csv("./PRJNA375935/PRJNA375935.tsv", sep='\t')