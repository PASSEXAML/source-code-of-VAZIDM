# 加载包
library(DESeq2)
library(EnhancedVolcano)

# 加载数据
raw_count_data <- read.csv("path/tao/raw_count_data.csv", row.names = 1)
sample_info <- read.csv("path/to/sample_info.csv")

# 确保样本ID一致
if (!all(colnames(raw_count_data) == sample_info$SampleID)) {
    stop("Sample IDs in count data and sample info do not match.")
}

# 创建DESeq数据集对象
dds <- DESeqDataSetFromMatrix(countData = raw_count_data, colData = sample_info, design = ~ Condition + Batch)

# 预处理数据（可选：移除低计数数据）
dds <- dds[rowSums(counts(dds)) > 1, ]

# 进行差异丰度分析
dds <- DESeq(dds)

# 提取结果
res <- results(dds, contrast = c("Condition", "treatment", "control"))
res <- res[order(res$padj), ]

# 筛选显著差异的基因
sig_res <- res[!is.na(res$padj) & res$padj < 0.05, ]

# 打印结果
print(head(sig_res))

# MA图
plotMA(res, main="DESeq2", ylim=c(-2, 2))

# 火山图
EnhancedVolcano(res,
    lab = rownames(res),
    x = 'log2FoldChange',
    y = 'pvalue',
    pCutoff = 0.05,
    FCcutoff = 1.0,
    title = 'Differential Abundance Analysis',
    subtitle = 'Treatment vs Control')
