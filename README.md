# ANN-Project
Weight Sparsification for Efficient Neural Network Compression and Hardware-Aware Optimization
# Weight Sparsification for Efficient Neural Network Compression and Hardware-Aware Optimization

This repository contains the code and experiments for a project on **weight sparsification** (pruning) applied to:

- **ResNet-18 on CIFAR-10** (≈ **11.2M** parameters)  
- **DistilGPT-2 on WikiText** (≈ **82M** parameters)

The goal is to study how different sparsity patterns affect:

- Accuracy (CIFAR-10) and perplexity (WikiText)  
- Model size and parameter count  
- **Inference latency**  
- **GPU memory usage**

All experiments were implemented in **PyTorch** and run on **Google Colab (paid tier)** using an **NVIDIA A100 GPU**.

Most numerical results are printed directly inside the Colab notebooks (`.ipynb`). If some values differ slightly from those reported in the written report, that is expected: training and pruning involve randomness (initialization, data order, etc.), and re-running full training to exactly match all numbers is expensive in terms of time and GPU usage.

---

## 1. Project Overview

Modern neural networks are heavily over-parameterized. Many weights contribute little to the final prediction, yet still incur memory, bandwidth, and energy cost.

This project compares:

- **Unstructured pruning**  
  Magnitude-based removal of individual weights (fine-grained, irregular sparsity).

- **Structured channel pruning**  
  Pruning entire convolutional filters/channels (coarse-grained, hardware-friendly sparsity).


The analysis focuses not only on *logical* sparsity (percentage of zero weights) but also on *hardware behavior*: latency and VRAM usage on an A100 GPU.






