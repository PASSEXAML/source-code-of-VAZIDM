from model.api import api
import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np

# 读取tsv文件
micro_data = pd.read_csv("./data/microbiome_external.tsv", sep='\t', index_col=0)

# 转置数据
micro_data = micro_data.transpose()
d = api();

# for i in np.arange(0.02, 0.06, 0.01):
d.dva(adata=micro_data, threads=1, W_v=0, W_x=10)
