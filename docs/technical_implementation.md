# Technical Implementation and Reproducibility

This document summarizes the software and hardware setup used for the audit, ensuring both **result fidelity** and **reproducibility** on high-performance computing environments.

---

## 1. Hardware Environment (RunPod)

The experiments were conducted on the infrastructure, optimized for deep learning workloads.

- **GPU:** NVIDIA A40 (48 GB VRAM)  
- **Access:** Remote SSH for real-time monitoring  
- **Optimizations:**
  - Multi-GPU support via DataParallel  
  - `torch.backends.cudnn.benchmark = True` for optimized convolution performance on Ampere architecture  

---

## 2. Software Architecture: TransUNet Focus

The file `src/baseline_method/model.py` implements a hybrid CNN-Transformer architecture.

### Encoder (CNN)

- Based on **EncoderBottleneck blocks**:
  - Conv 1×1 → Conv 3×3 → Conv 1×1  
  - Batch normalization applied at each stage  

### Transformer Bottleneck (ViT)

- Multi-Head Attention: **4 heads**  
- Depth: **8 transformer layers**  
- Embedding dimension: **1024**  

### Decoder

- DecoderBottleneck blocks with:
  - Bilinear upsampling  
  - Skip connections from encoder  
- Objective: restore spatial resolution and refine segmentation maps  

---

## 3. Training Protocol (`train.py`)

The training pipeline is designed to remain deterministic, even under distributed computation.

---

### A. Reproducibility

A global seed is enforced:

```python
set_seed(42)
```

This ensures reproducibility across:

- Python random generators  
- NumPy  
- PyTorch (CPU and CUDA)  
- DataLoader workers (`worker_init_fn`)  

---

### B. Optimization Strategy

- **Optimizer:** SGD  
- **Momentum:** 0.9  
- **Weight Decay:** 1e-4  

- **Learning Rate Scheduler:** Polynomial decay  

```text
lr = base_lr × (1 - iter / max_iter)^0.9
```

---

### C. Loss Function

Hybrid loss combining pixel-wise and structural objectives:

```text
Loss = 0.5 × CrossEntropy + 0.5 × DiceLoss
```

- CrossEntropy → pixel accuracy  
- DiceLoss → global shape consistency  

---

## 4. Preprocessing Pipeline (`SimpleDataset`)

The data loading pipeline follows strict rules to prevent unintended data leakage.

- **Normalization:**
  - Pixel values scaled to **[0, 1]**

- **Augmentation (Train only):**
  - Random rotations (±20°)  
  - Horizontal and vertical flips  

- **Resizing:**
  - Images: **224×224** (bilinear interpolation)  
  - Masks: **224×224** (nearest neighbor interpolation)  

**Rationale:**

- Bilinear interpolation ensures smooth image scaling  
- Nearest neighbor preserves binary label integrity  

---
