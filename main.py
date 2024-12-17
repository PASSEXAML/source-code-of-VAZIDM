from model.api import api
import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np

file_path = "E:\\work_code\\source-code-of-VAZIDM\\result\\simulated_3"

# 读取tsv文件
micro_data = pd.read_csv("./data/simulated_3/simulated3.tsv", sep='\t', index_col=0)

# 转置数据
micro_data = micro_data.transpose()
d = api();

d.vazidm(adata=micro_data, threads=1, file_path=file_path)
