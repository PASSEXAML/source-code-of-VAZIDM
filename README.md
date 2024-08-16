# VAZIDM
**VAZIDM** is a model that combines **Variational Autoencoder (VAE)** and **Generative Adversarial Network (GAN)** techniques, along with a **Zero-Inflated Negative Binomial (ZINB)** model, to handle sparse and complex microbiome data effectively.

# Description
Due to uneven sequencing depth and the inherent diversity and complexity of microbial communities, microbiome data often exhibit high sparsity, with a large number of zeros present in the data. This poses challenges for accurate downstream analysis.

In this study, we developed the **Variational Adversarial Zero-Inflated Denoising Model (VAZIDM)**, which combines the probabilistic generation capability of a **Variational Autoencoder (VAE)** with the adversarial training mechanism of a **Generative Adversarial Network (GAN)** to capture the complex and nonlinear features of microbiome sequencing data. Additionally, VAZIDM uses a **Zero-Inflated Negative Binomial (ZINB)** model to fit count data that contain a large number of zeros and exhibit excessive dispersion.

To evaluate VAZIDM against traditional autoencoder and denoising models, we conducted multiple experimental assessments using **16S rRNA sequencing data** and **whole-metagenome sequencing (WMS)** data. The experimental results demonstrate that VAZIDM exhibits significant advantages over traditional autoencoder and denoising models across several key evaluation metrics.

# Usage
You can run the model by tutorial.ipynb



where matrix.csv is a CSV/TSV-formatted raw count matrix with genes in rows and cells in columns. Cell and gene labels are mandatory.
