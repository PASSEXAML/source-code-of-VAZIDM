from model.api import api
import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np

file_path = "E:\\work_code\\Dva_ana\\result\\new_PRJEB13870"

# 读取tsv文件
micro_data = pd.read_csv("./data/PRJEB13870/PRJEB13870.tsv", sep='\t', index_col=0)

# 转置数据
micro_data = micro_data.transpose()
d = api();

d.dva(adata=micro_data, threads=1, file_path=file_path)
