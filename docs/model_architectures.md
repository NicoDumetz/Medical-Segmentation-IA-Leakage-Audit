# Model Architectures and Training Strategies

This document details the technical specifications of the two TransUNet architecture variants compared in this audit.  
The objective is to isolate the impact of structural design on overfitting sensitivity.

---

## 1. Architecture Comparison

| Feature                     | baseline_method (GitHub mkara44)        | proposed_robust_method (Paper Standard) |
|----------------------------|-----------------------------------------|------------------------------------------|
| Origin                     | mkara44/transunet_pytorch               | Academic implementation                  |
| CNN Backbone               | Custom CNN (3 bottlenecks)              | Hybrid ResNetV2-50                       |
| Feature Map                | 14×14 (progressive convolution)         | 14×14 (patch embedding on CNN features)  |
| Transformer                | 8 layers / 4 heads / dim 1024           | 12 layers / 12 heads / dim 768           |
| Decoder                    | U-Net (standard upsampling)             | DecoderCup (cascaded upsampler)          |
| Skip Connections           | 4 CNN → Decoder                         | 3 CNN → Decoder                          |

---

### Baseline Model Specifics

The implementation from
relies on a more direct CNN encoding strategy.

- The Vision Transformer (ViT) operates on CNN feature outputs treated as token grids  
- Although `patch_size=16` is defined, patch extraction is implicit via CNN features  
- Embedding dimension: **1024 with only 4 attention heads**

**Implication:**

- High embedding dimensionality + low number of heads  
- Encourages **dense feature representations**
- In presence of data leakage, this increases the model’s ability to **memorize local textures** rather than learn generalizable features

---

### Proposed Robust Model Specifics

This variant follows the standard **ViT-B/16 configuration** described by :

- Hidden size: **768**
- MLP dimension: **3072**
- Transformer depth: **12 layers / 12 heads**
- Decoder: **DecoderCup**
  - Uses convolutional refinement after upsampling
  - Improves segmentation boundary precision

**Implication:**

- More balanced attention distribution  
- Better inductive bias for **global + local feature integration**  
- Reduced tendency toward memorization compared to the baseline variant  

---

## 2. Training Protocol and Pipeline

To ensure fairness, both models share an identical simplified training pipeline based on the baseline repository.

---

### A. Preprocessing and Image Processing

- **Resizing:**  
  All images are resized to **224×224** using OpenCV (`INTER_LINEAR`)  

  **Consequence:**  
  - Original aspect ratio is not preserved  
  - Anatomical proportions may be distorted  

- **Normalization:**  
  Pixel values scaled to **[0, 1]**

- **Data Augmentation (Train only):**
  - Random rotations  
  - Horizontal and vertical flips  

  **Limitation:**  
  These simple geometric transformations do not break the **temporal correlation** between consecutive ultrasound frames.

---

### B. Optimizer Parameters

- **Optimizer:** SGD (Stochastic Gradient Descent)  
- **Momentum:** 0.9  
- **Weight Decay:** 1e-4  
- **Initial Learning Rate:** 0.01  

- **Learning Rate Scheduler:** Polynomial decay applied per iteration  

---

## 3. Audit Criterion: Absence of Pretraining

A critical constraint in this audit is that both models are trained **from scratch**:

- No ImageNet pretrained weights for CNN backbones  
- No pretrained ViT weights for Transformer layers  

---

### Analysis

Training from scratch on moderately sized medical datasets has important implications:

- The model becomes highly sensitive to **repetitive patterns**
- In a **Frame-Mix split**, it exploits:
  - Pixel-level similarities between adjacent frames  
  - Stable acquisition artifacts  

Instead of learning robust semantic representations, the model:

> Learns shortcut mappings driven by spatial redundancy

This effect amplifies the impact of **data leakage**, leading to artificially inflated performance metrics.

---
