# Does Accuracy Stability Under Compression Mean Representational Stability?

When you compress a neural network and accuracy barely changes, the standard 
conclusion is that the compression "worked." This project questions that conclusion.

Using Centered Kernel Alignment (CKA) and Grad-CAM, we probe whether compressed 
models maintain the same internal representations as their full-precision counterparts
— or whether they arrive at the same accuracy via structurally different computational 
pathways.

This distinction matters. A model that solves the same task differently may fail 
differently, attend to different image features, and behave unpredictably under 
distribution shift even when benchmark accuracy looks identical.

## Motivation

Han et al. (2016) showed that pruning + quantisation can reduce model size by 
orders of magnitude with <1% accuracy drop. Every compression paper since reports 
the same metric: accuracy before and after.

What they don't report is what changed inside.

Kornblith et al. (2019) introduced CKA specifically to compare representations 
across networks and training runs. It hasn't been applied to the compression 
transition. This project does that.

## Research Question

> When a pruned or quantised ResNet-18 maintains near-identical classification 
> accuracy on CIFAR-10, does it also maintain near-identical layer-wise 
> representations — or does it restructure its internal computation to compensate?

## What is Done

- Establish a fine-tuned ResNet-18 baseline on CIFAR-10
- Apply three compression strategies: magnitude pruning (30/50/70% sparsity), 
  post-training INT8 quantisation, and knowledge distillation to MobileNetV2
- Measure layer-wise CKA similarity between original and each compressed model
- Compare Grad-CAM attention maps on matched correct/incorrect predictions
- Report: accuracy vs representational similarity tradeoff curves

## Key References

- Han et al. (2016). Deep Compression. ICLR.
- Kornblith et al. (2019). Similarity of Neural Network Representations. ICML.
- Sharkey et al. (2025). Open Problems in Mechanistic Interpretability. arXiv.

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/compression-representation-analysis.git
cd compression-representation-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Structure
```
notebooks/
  01_baseline.ipynb          # Fine-tune ResNet-18, establish baseline accuracy
  02_compression.ipynb       # Apply pruning, quantisation, distillation
  03_cka_analysis.ipynb      # Layer-wise CKA comparison
  04_gradcam.ipynb           # Attention map comparison
src/
  cka.py                     # CKA implementation
  compression.py             # Compression utilities
  gradcam.py                 # Grad-CAM implementation
results/
  figures/                   # All plots
  tables/                    # Accuracy and CKA results
```

## Status

Work in progress. Notebooks added as experiments complete.