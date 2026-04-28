# Medical Segmentation IA Leakage Audit and Clinical Risk

## Overview

This repository presents a **systematic audit of data leakage in medical image segmentation**, focusing on **Crohn’s disease ultrasound segmentation** using TransUNet architectures.

The objective is to expose a critical issue in medical AI:

> **Improper data splitting can generate artificially high metrics that do not reflect real clinical performance and introduce significant clinical risk.**

This work demonstrates how **Frame-Mix leakage** leads to misleading conclusions, while **Patient-Aware splitting** provides reliable and clinically meaningful evaluation.

---

## Documentation and Reports

### Documentation

- [Problem Statement](docs/problem_statement.md)
- [Splitting Protocol](docs/splitting_protocol.md)
- [Model Architectures](docs/model_architectures.md)
- [Technical Implementation](docs/technical_implementation.md)
- [Metrics and Evaluation](docs/metrics_and_evalutation.md)
- [Global Clinical Risk Report](docs/global_clinical_risk_report.md)

### Experiment Reports

- [Baseline Frame-Mix Summary](results/baseline_results/frame_mix_protocol/summary_report.md)
- [Baseline Patient-Aware Summary](results/baseline_results/patient_aware_protocol/summary_report.md)
- [Baseline Comparison Summary](results/baseline_results/baseline_comparison_summary.md)
- [Robust Frame-Mix Summary](results/robust_result/frame_mix_protocol/summary_report.md)
- [Robust Patient-Aware Summary](results/robust_result/patient_aware_protocol/summary_report.md)
- [Robust Comparison Summary](results/robust_result/robust_summary_comparison.md)

---

## Repository Structure

```
.
├── docs
├── src
├── results
│   ├── baseline_results
│   └── robust_result
└── requirement.txt
```

---

## Experimental Design

Two architectures were evaluated:

- **Baseline TransUNet**
- **Robust TransUNet (ResNetV2 + ViT)**

Each model was trained under two protocols:

### 1. Frame-Mix (Biased)

- Frames from the same patient appear across train/val/test  
- Introduces **data leakage**  
- Leads to **inflated and misleading metrics**

### 2. Patient-Aware (Strict)

- Full separation at **patient level**  
- No overlap between splits  
- Reflects **real clinical deployment conditions**

---

## Results Organization

All experimental outputs are centralized in the `results/` directory.

### Baseline Model

- Frame-Mix protocol  
  → `results/baseline_results/frame_mix_protocol/`

- Patient-Aware protocol  
  → `results/baseline_results/patient_aware_protocol/`

- Global comparison  
  → `results/baseline_results/baseline_comparison_summary.md`

---

### Robust Model (Paper Implementation)

- Frame-Mix protocol  
  → `results/robust_result/frame_mix_protocol/`

- Patient-Aware protocol  
  → `results/robust_result/patient_aware_protocol/`

- Global comparison  
  → `results/robust_result/robust_summary_comparison.md`

---

## Key Findings

- **Data leakage introduces massive bias**
  - Up to **+20 Dice points artificially gained**

- **Model capacity amplifies the illusion**
  - Larger models overfit more efficiently under leakage

- **True generalization remains limited**
  - Real-world Dice consistently ~**65–73%**

- **Patient-Aware split is the only valid evaluation**
  - Internal metrics align with clinical reality

---

## Clinical Risk

A model validated under leakage conditions:

- Reports very high performance (>90% Dice)  
- Maintains high precision (>90%)  
- But suffers from **low recall (~55–60%)**

### Consequences

- Large portions of pathological regions are **missed**  
- Lesion extent is **systematically underestimated**  
- The model appears reliable but is **clinically unsafe**

> Up to **40% of lesions may remain undetected** on unseen patients.

---

## Important Note

All models exhibit some degree of overfitting due to:

- Limited dataset size  
- Lack of strong data augmentation  
- No domain-specific pretraining  

However:

> **This does not alter the conclusions.**

- Frame-Mix hides overfitting and inflates performance  
- Patient-Aware reveals true generalization behavior  

---

## Installation

```
pip install -r requirement.txt
```

---

## Running Experiments

Example:

```
DATASET_ROOT=/path/to/dataset python3 src/model/train.py
```

---

## Final Statement

> The issue is not model architecture — it is evaluation methodology.

Any medical AI model evaluated without strict patient-level separation:

- Produces misleading metrics  
- Fails to generalize  
- Introduces **real clinical risk**

---
