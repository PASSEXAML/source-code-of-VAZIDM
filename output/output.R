   library(ggplot2)
   library(reshape2)  # 加载reshape2包
   path = "zinb_loss"
   vs_path = "vs_loss"
   mean_path = "output/mean.tsv"
   originalData_path = "data/microbiome_external.tsv"
   denoisedData_path = paste(path, "/output_values.tsv", sep = "")
   vs_path = paste(vs_path, "/output_values.tsv", sep = "")
   original_data = read.delim(originalData_path)
   denoised_data = read.delim(denoisedData_path)
   vs_data = read.delim(vs_path)
   mean_data = read.delim(mean_path)

   colnames_data = colnames(denoised_data)
   denoised_data = as.matrix(denoised_data)
   vs_data = as.matrix(vs_data)
   mean_data = as.matrix(mean_data)

## Compositional normalization analysis (alpha diversity)
   diversity <- function (x) {
      # shannon index
  # Ensure x is numeric
  if (!is.numeric(x)) {
    x <- as.numeric(x)
  }

  # Replace NA and NaN values with 0
  x[!is.finite(x)] <- 0
     x[x == 0] <- .Machine$double.xmin
  # shannon index
  sh_ele <- -x * log(x)

  # shannon index sum
  sh <- sum(sh_ele)

  # simpson index
  sp <- 1- sum(x*x)

  return (list(Shannon = sh, Simpson = sp))
   }


   div <- list()
   div1 <- list()
   div2 <- list()
   div3 <- list()
   Shannon_val <- c()
   Simpson_val <- c()
   Shannon_val1 <- c()
   Simpson_val1 <- c()
   Shannon_val2 <- c()
   Simpson_val2 <- c()
   Shannon_val3 <- c()
   Simpson_val3 <- c()

   mean_n <- ncol(mean_data)
   for (i in 1:mean_n) {
     div3[[i]] <- diversity(mean_data[, i])
     Shannon_val3[i] <- div3[[i]]$Shannon
     Simpson_val3[i] <- div3[[i]]$Simpson
   }

   ae_n <- ncol(vs_data)
   for (i in 1:ae_n) {
     div2[[i]] <- diversity(vs_data[, i])
     Shannon_val2[i] <- div2[[i]]$Shannon
     Simpson_val2[i] <- div2[[i]]$Simpson
   }

   n <- ncol(denoised_data)
   for(i in 1:n){
     column_name = colnames_data[i]
     select_data = original_data[,column_name]
     div1[[i]] <- diversity(select_data)
     Shannon_val1[i] <- div1[[i]]$Shannon
     Simpson_val1[i] <- div1[[i]]$Simpson
   }


   for(i in 1:n) {
      div[[i]] <- diversity(denoised_data[,i])
      Shannon_val[i] <- div[[i]]$Shannon
      Simpson_val[i] <- div[[i]]$Simpson
   }


   # Save the results
   # Create a data frame
   df <- data.frame(Sample = 1:length(Shannon_val), Shannon_Index = Shannon_val, Shannon_Index1 = Shannon_val1, Shannon_Index2 = Shannon_val2, Shannon_val3 = Shannon_val3)
   # Use ggplot2 to create the plot
   ggplot(df, aes(x = Sample, y = Shannon_Index)) +
     geom_line(aes(y= Shannon_Index1), color = "orange", size = 1) +
     geom_point(aes(y= Shannon_Index1), color = "orange", size = 4, shape = 2, stroke = 2) +
     geom_line(aes(y= Shannon_Index2), color = "red", size = 1) +
     geom_point(aes(y= Shannon_Index2), color = "red", size = 4, shape = 3, stroke = 2) +
     geom_line(aes(y= Shannon_val3), color = "green", size = 1) +
     geom_point(aes(y= Shannon_val3), color = "green", size = 4, shape = 4, stroke = 2) +
   geom_line(color = "#52cbe9", size = 1) +
   geom_point(color = "#52cbe9", size = 4, shape = 1, stroke = 2) +
   labs(x = "Sample", y = "Shannon Index") +
   theme_minimal() +
     theme(plot.background = element_rect(fill = "white"),
           panel.grid.major = element_line(colour = "grey"),
           panel.grid.minor = element_line(colour = "grey"))
   # Save the plot
   ggsave(paste(path, "/Shannon_Index.png", sep = ""))

   df1 <- data.frame(Sample = 1:length(Simpson_val), Simpson_Index = Simpson_val, Simpson_Index1 = Simpson_val1, Simpson_Index2 = Simpson_val2,Simpson_val3 = Simpson_val3)
   df1_melt <- melt(df1, id.vars = "Sample")
   ggplot(df1, aes(x = Sample, y = Simpson_Index)) +
     geom_line(aes(y= Simpson_Index1), color = "orange", size = 1) +
     geom_point(aes(y= Simpson_Index1), color = "orange", size = 4, shape = 2, stroke = 2) +
     geom_line(aes(y= Simpson_Index2), color = "red", size = 1) +
     geom_point(aes(y= Simpson_Index2), color = "red", size = 4, shape = 3, stroke = 2) +
     geom_line(aes(y= Simpson_val3), color = "green", size = 1) +
     geom_point(aes(y= Simpson_val3), color = "green", size = 4, shape = 4, stroke = 2) +
   geom_line(color = "#52cbe9", size = 1) +
   geom_point(color = "#52cbe9", size = 4, shape = 1, stroke = 2) +
   labs(x = "Sample", y = "Simpson Index") +
   theme_minimal() +
     theme(plot.background = element_rect(fill = "white"),
           panel.grid.major = element_line(colour = "grey"),
           panel.grid.minor = element_line(colour = "grey"))
   # Save the plot
   ggsave(paste(path, "/Simpson_Index.png", sep = ""))

   #DA analysis

