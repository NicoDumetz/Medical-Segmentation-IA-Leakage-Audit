# Audit Report: Frame-Mix Protocol on "Robust" TransUNet (Paper Implementation)

---

## 1. Illusion of Performance: Peak Metrics Summary

The **standard TransUNet architecture (ResNetV2 + ViT)**, due to its higher capacity, demonstrates an exceptional ability to memorize the training distribution when exposed to a **Frame-Mix (leakage-prone) split**.

| Evaluation | Epoch 50 | Epoch 40 | Epoch 30 |
|------------|----------|----------|----------|
| Train Dice | 95.13%   | 94.00%   | 92.65%   |
| Val Dice   | 93.36%   | 93.16%   | 92.60%   |
| Internal Test Dice | 93.29% (thr=0.4) | 93.07% (thr=0.5) | 92.45% (thr=0.5) |

### Dice Evolution

![Dice Curve](curves/Dice.png)

### Loss Evolution

![Loss Curve](curves/Loss.png)

### Interpretation

On internal evaluation, the model appears near-perfect:

- Dice scores exceed **93%**
- Training and validation curves remain tightly coupled
- No apparent overfitting signal

A superficial analysis would conclude that the architecture successfully solves Crohn’s disease segmentation.

---

## 2. Clinical Reality Check: Independent Holdout

Evaluation on **fully unseen patients** reveals a severe performance collapse.

### Independent Metrics

![Independent Dice/IoU](curves/independant_iou_dice.png)

![Independent Precision/Recall](curves/independant_rec_prec.png)

| Epoch | Dice (Holdout) | Δ vs Internal Test | Recall | Precision |
|------|----------------|--------------------|--------|-----------|
| 30   | 73.66%         | -18.79%            | 62.63% | 89.56%    |
| 40   | 72.24%         | -20.83%            | 61.22% | 90.09%    |
| 50   | 70.17%         | -23.12%            | 57.88% | 91.03%    |

### Key Observation

- Precision remains **very high (>90%)**
- Recall drops significantly (**~58–62%**)

### Interpretation

The model does not generate false positives, but fails to detect a substantial portion of pathological regions.

> Up to **40% of lesions remain undetected** on new patients.

---

## 3. Evidence of Memorization: Epoch Paradox

### Threshold Optimization

![Dice Threshold](threshold_analysis/Dice_threshold.png)

The most critical observation lies in the **divergence between internal and external performance across epochs**:

- Internal Test (biased):
  - Improves from **92.45% → 93.29%**
- Independent Holdout:
  - Degrades from **73.66% → 70.17%**

### Interpretation

Between Epoch 30 and Epoch 50:

- The model stops learning generalizable patterns
- It reallocates capacity to memorize dataset-specific features:
  - Ultrasound texture
  - Acquisition artifacts
  - Patient-specific anatomy

This leads to:

- Improved performance on corrupted test data
- Degraded performance on real-world data

This behavior is a direct signature of **data leakage exploitation**.

---

## 4. Bias Amplification Through Model Capacity

Comparison with the baseline TransUNet under the same Frame-Mix protocol:

| Model | Internal Dice | Holdout Dice | Delta |
|-------|--------------|--------------|--------|
| Baseline | ~86% | ~69% | -16.73% |
| Robust Model | ~93% | ~70% | -23.12% |

### Interpretation

Increasing model capacity does not mitigate bias:

- It **amplifies it**
- The model becomes more efficient at memorization
- The gap between internal and real performance widens

---

## 5. Final Assessment

The apparent superiority of the robust architecture is an **artifact of evaluation bias**.

- Internal performance: inflated by leakage
- Real performance: constrained by generalization limits

### True Performance Estimate

> **~73.66% Dice (Epoch 30)** represents the best generalization point before overfitting dominates

---

## 6. Auditor Conclusion

Using a high-capacity model (ResNetV2 + Transformer) on a dataset affected by data leakage:

- Does not correct the bias  
- **Exacerbates it**

### Final Insight

- The model learns:
  - Dataset-specific signatures  
- Instead of:
  - Generalizable disease features  

This results in:

> A stronger illusion of performance and a more severe clinical failure

---
