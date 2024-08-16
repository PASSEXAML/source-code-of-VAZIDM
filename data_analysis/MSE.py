import pandas as pd
from sklearn.metrics import mean_squared_error

# 路径可能需要根据您的文件位置进行调整
original_data_path = '../data/IBD_PRISM/microbiome_PRISM.tsv'
predicted_data_path = '../data/IBD_PRISM/output_values.tsv'

# 加载原始数据和预测数据
original_data = pd.read_csv(original_data_path, sep='\t')
predicted_data = pd.read_csv(predicted_data_path, sep='\t')

# 确定两个数据集中共有的样本列
common_samples = original_data.columns.intersection(predicted_data.columns)

# 提取共有样本的数据进行比较
original_data_common = original_data[common_samples]
predicted_data_common = predicted_data[common_samples]

# 计算每个共有样本的MSE
mse_values = {}
for sample in original_data_common.columns[1:]:  # 排除第一列（通常为标签或描述列）
    mse = mean_squared_error(original_data_common[sample], predicted_data_common[sample])
    mse_values[sample] = mse

# 将MSE值转换为DataFrame
mse_df = pd.DataFrame(list(mse_values.items()), columns=['Sample', 'MSE'])
# 保存MSE值
mse_df.to_csv('../result/IBD_PRISM/MSE/vae_values.tsv', sep='\t', index=False)
