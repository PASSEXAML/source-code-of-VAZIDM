# VAZIDM
**VAZIDM** is a model that combines **Variational Autoencoder (VAE)** and **Generative Adversarial Network (GAN)** techniques, along with a **Zero-Inflated Negative Binomial (ZINB)** model, to handle sparse and complex microbiome data effectively.

# Description
Due to uneven sequencing depth and the inherent diversity and complexity of microbial communities, microbiome data often exhibit high sparsity, with a large number of zeros present in the data. This poses challenges for accurate downstream analysis.

In this study, we developed the **Variational Adversarial Zero-Inflated Denoising Model (VAZIDM)**, which combines the probabilistic generation capability of a **Variational Autoencoder (VAE)** with the adversarial training mechanism of a **Generative Adversarial Network (GAN)** to capture the complex and nonlinear features of microbiome sequencing data. Additionally, VAZIDM uses a **Zero-Inflated Negative Binomial (ZINB)** model to fit count data that contain a large number of zeros and exhibit excessive dispersion.

To evaluate VAZIDM against traditional autoencoder and denoising models, we conducted multiple experimental assessments using **16S rRNA sequencing data** and **whole-metagenome sequencing (WMS)** data. The experimental results demonstrate that VAZIDM exhibits significant advantages over traditional autoencoder and denoising models across several key evaluation metrics.

# Usage
You can run the model by tutorial.ipynb

```python
from model.api import api  # Importing a custom API module
import pandas as pd        # Importing pandas for data manipulation
import tensorflow as tf    # Importing TensorFlow (not used in the script)
import numpy as np         # Importing NumPy (not used in the script)

result_path = "result\\..."  # Defining the path to save the results

# Read the TSV file into a pandas DataFrame
micro_data = pd.read_csv(".../.../matrix.tsv", sep='\t', index_col=0)

# Transpose the DataFrame (switch rows and columns)
micro_data = micro_data.transpose()

# Initialize the API object
d = api()

# Apply the 'dva' function on the transposed data with specified parameters
d.dva(adata=micro_data, threads=1, file_path=result_path)
```


where matrix.csv is a TSV-formatted raw count matrix with microbiomes in rows and samples in columns.
