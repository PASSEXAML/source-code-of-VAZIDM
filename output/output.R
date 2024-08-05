library(ggplot2)
library(reshape2)  # 加载reshape2包

# 文件路径
path <- "E:/work_code/Dva_ana"

ae_path <- paste(path, "/data/PRJEB13870/mean.tsv", sep = "")
originalData_path <- paste(path, "/data/PRJEB13870/PRJEB13870.tsv", sep = "")
vae_path <- paste(path, "/data/PRJEB13870/output_values.tsv", sep = "")
gan_path <- paste(path, "/result/PRJEB13870/output_values.tsv", sep = "")

# 读取数据
ae_data <- read.delim(ae_path)
original_data <- read.delim(originalData_path)
gan_data <- read.delim(gan_path)
vae_data <- read.delim(vae_path)

# 删除第一列
# 删除第一列
ae_data <- ae_data[,-1]
original_data <- original_data[,-1]
gan_data <- gan_data[,-1]
vae_data <- vae_data[,-1]

colnames_data <- colnames(gan_data)
gan_data <- as.matrix(gan_data)
ae_data <- as.matrix(ae_data)
vae_data <- as.matrix(vae_data)

# 计算Shannon指数和Simpson指数的函数
diversity <- function(x) {
  if (!is.numeric(x)) {
    x <- as.numeric(x)
  }

  x[!is.finite(x)] <- 0
  x[x == 0] <- .Machine$double.xmin

  sh_ele <- -x * log(x)
  sh <- sum(sh_ele)

  sp <- 1 - sum(x * x)

  return(list(Shannon = sh, Simpson = sp))
}

# 计算Shannon指数和Simpson指数
div1 <- list()
div2 <- list()
div3 <- list()
div4 <- list()
Shannon_val1 <- c()
Simpson_val1 <- c()
Shannon_val2 <- c()
Simpson_val2 <- c()
Shannon_val3 <- c()
Simpson_val3 <- c()
Shannon_val4 <- c()
Simpson_val4 <- c()

vae_n <- ncol(vae_data)
for (i in 1:vae_n) {
  div4[[i]] <- diversity(vae_data[, i])
  Shannon_val4[i] <- div4[[i]]$Shannon
  Simpson_val4[i] <- div4[[i]]$Simpson
}

# 计算AE数据的指数
ae_n <- ncol(ae_data)
for (i in 1:ae_n) {
  div3[[i]] <- diversity(ae_data[, i])
  Shannon_val3[i] <- div3[[i]]$Shannon
  Simpson_val3[i] <- div3[[i]]$Simpson
}

# 计算原始数据的指数
n <- ncol(gan_data)
for (i in 1:n) {
  column_name <- colnames_data[i]
  select_data <- original_data[, column_name]
  div2[[i]] <- diversity(select_data)
  Shannon_val2[i] <- div2[[i]]$Shannon
  Simpson_val2[i] <- div2[[i]]$Simpson
}

# 计算去噪数据的指数
for (i in 1:n) {
  div1[[i]] <- diversity(gan_data[, i])
  Shannon_val1[i] <- div1[[i]]$Shannon
  Simpson_val1[i] <- div1[[i]]$Simpson
}

# 准备Shannon指数的绘图数据
plot_data <- data.frame(
  Sample = rep(1:n, 4),
  Shannon = c(Shannon_val1, Shannon_val2, Shannon_val3, Shannon_val4),
  Group = rep(c("VAE_GAN", "Original", "AE", "VAE"), each = n)
)

# 绘制Shannon指数折线图
p1 <- ggplot(plot_data, aes(x = Sample, y = Shannon, color = Group, shape = Group)) +
  geom_line(aes(linetype = Group), linewidth = 1) +
  geom_point(size = 3) +
  labs(title = "Shannon Diversity Index",
       x = "Sample",
       y = "Shannon Index") +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid")) +  # 设置不同的线型
  scale_color_manual(values = c("pink", "#6fb4f9", "#f8c666", "red")) +  # 设置不同的颜色
  scale_shape_manual(values = c(16, 17, 18, 19)) +  # 设置不同的标志形状
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"),
        panel.grid.major = element_line(colour = "grey"),
        panel.grid.minor = element_line(colour = "grey"),
        legend.title = element_blank(),  # 移除图例标题
        legend.position = "bottom",      # 将图例放在底部
        legend.box = "horizontal")       # 设置图例水平排列

# 保存Shannon指数折线图
ggsave(filename = paste(path, "/result/PRJEB13870/Shannon_Index.png", sep = ""), plot = p1, width = 10, height = 7, dpi = 300)

# 准备Simpson指数的绘图数据
plot_data1 <- data.frame(
  Sample = rep(1:n, 4),
  Simpson = c(Simpson_val1, Simpson_val2, Simpson_val3, Simpson_val4),
  Group = rep(c("VAE_GAN", "Original", "AE", "VAE"), each = n)
)

# 绘制Simpson指数折线图
p2 <- ggplot(plot_data1, aes(x = Sample, y = Simpson, color = Group, shape = Group)) +
  geom_line(aes(linetype = Group), linewidth = 1) +
  geom_point(size = 3) +
  labs(title = "Simpson Diversity Index",
       x = "Sample",
       y = "Simpson Index") +
  scale_linetype_manual(values = c("solid", "solid", "solid", "solid")) +  # 设置不同的线型
  scale_color_manual(values = c("pink", "#6fb4f9", "#f8c666", "red")) +  # 设置不同的颜色
  scale_shape_manual(values = c(16, 17, 18, 19)) +  # 设置不同的标志形状
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"),
        panel.grid.major = element_line(colour = "grey"),
        panel.grid.minor = element_line(colour = "grey"),
        legend.title = element_blank(),  # 移除图例标题
        legend.position = "bottom",      # 将图例放在底部
        legend.box = "horizontal")       # 设置图例水平排列

# 保存Simpson指数折线图
ggsave(filename = paste(path, "/result/PRJEB13870/Simpson_Index.png", sep = ""), plot = p2, width = 10, height = 7, dpi = 300)
