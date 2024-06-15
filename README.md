# Optimising Visual Input for Cortical Neurons: A CNN-based MEI Approach Compared to Gabor Patches

## Introduction

This project leverages deep learning techniques to optimize visual stimuli for cortical neurons, focusing on generating the Most Excitatory Input (MEI) for a single neuron in the visual system of a mouse. Using end-to-end trained convolutional neural networks (CNNs), we aim to construct stimuli that maximally activate neurons, and compare the efficacy of these stimuli to traditional Gabor patches.

## Dataset Overview

The dataset comprises neural activity recordings from mouse visual cortex neurons exposed to natural scenes. The data includes:

- Neural activity raster maps for three distinct time slots.
- Analysis of neural responses in terms of correlation, entropy, and variance.
- An unsupervised anomaly detection algorithm, Isolation Forest, to identify the most informative neurons.

## Methodology

### Most Important Neurons (MINs) Selection

We used oracle correlation methods to identify neurons with high information content. The process involves:

1. Constructing RasterMaps to visualize neural activity.
2. Analyzing neural response patterns using correlation, entropy, and variance metrics.
3. Applying Isolation Forest for anomaly detection.
4. Selecting neurons with high oracle correlations and variance (above the 60th and 50th percentiles, respectively).

### Convolutional Neural Network (CNN)

Our CNN architecture consists of three core blocks followed by a neuron-specific readout layer:

#### Core Blocks

1. **Convolutional Layer**: Kernel size matches available Gabor filters.
2. **Batch Normalization**: Prevents overfitting by normalizing inputs.
3. **Non-linearity Activation**: $ELU$ (Exponential Linear Unit) function
4. **Pooling Layer**: Applied in the first block to enhance performance by downsampling feature maps.

#### Readout Layer

Transforms extracted features into a scalar representing the average spike activity of a neuron using an affine transformation followed by the activation function $ELU(x) +1$

### Model Training

We trained our CNN using the following metrics:

- **Loss**: Monitored during training to ensure convergence.
- **Validation Correlation**: Highest-performing neuron achieved a validation correlation of $62$%.

### Most Excitatory Input (MEI) Generation

The MEI generation algorithm iteratively optimizes an image to maximize neural activation:

```python
I_MEI = WhiteNoiseImage()
d_c = DecreaseConstant()
n_steps = 1000
sigma = 7

for i in range(n_steps):
    grads = grad(CNN(I_MEI))
    fftgrad = fftsmooth(grads)
    I_MEI = I_MEI + alpha * fftgrad
    I_MEI = gaussblur(I_MEI, sigma)
    sigma -= d_c
    I_MEI = scale(I_MEI, 0, 1)

return I_MEI
```

This process involves gradient ascent steps combined with Fourier Transform smoothing and Gaussian blurring to refine the image.

### Results 
- **High-Performing Neurons**: Selection based on oracle correlations and subsequent performance in the CNN.
- **MEI Generation**: Algorithm robustness confirmed through stable neural responses and standard deviation trends.

### Gabor Comparison 
Gabor filters, mathematical models mimicking visual cortex neuron responses, were compared to MEIs using metrics like luminance, contrast, and symmetry indices. Despite structural differences, both stimuli exhibited similar metric values, suggesting shared characteristics in driving neuronal responses.

### Conclusions and Limitations
Our analysis confirms that MEIs can elicit stronger neural responses than Gabor patches, aligning with findings by Walker et al. (2019). However, in vivo validation is necessary for further confirmation. Limitations include a small dataset and computational constraints leading to image resizing. Future research could expand to diverse cortical regions and multiple neuron predictions to enhance simulation sophistication.

### Bibliography
[Walker, et al., Nat. Neurosci., 2019](https://www.nature.com/articles/s41593-019-0517-x)

[Guidelines for the Computational Analysis of Single-Cell RNA Sequencing Data](https://www.nature.com/articles/s41596-020-00409-w)

[Best Practices for Single-Cell Analysis Across Modalities](https://www.nature.com/articles/s41576-023-00586-w)

[Current Best Practices in Single-Cell RNA-seq Analysis: A Tutorial](https://www.embopress.org/doi/full/10.15252/msb.20188746)


