# The Power Trap in Medical AI: Robust TransUNet Under Frame-Mix and Patient-Aware Protocols

This report analyzes the behavior of a **high-capacity TransUNet architecture (ResNetV2 + ViT)** under two data partitioning strategies:

- **Frame-Mix (leakage-prone)**
- **Patient-Aware (strict, clinically valid)**

The objective is to evaluate how the same model behaves under biased vs clinically sound evaluation conditions.

---

## 1. Performance Summary (Epoch 50)

| Configuration | Internal Test Dice | Independent Holdout Dice | Delta (Test → Holdout) |
|--------------|------------------|--------------------------|------------------------|
| Frame-Mix (Biased) | 93.29% | 70.17% | -23.12% |
| Patient-Aware (Strict) | 67.70% | 69.59% | +1.89% |

### Key Observation

- Frame-Mix produces **extremely high internal performance**
- Patient-Aware produces **lower but stable and realistic performance**
- Only Patient-Aware provides **reliable estimation of real-world performance**

---

## 2. Frame-Mix Behavior: Over-Optimistic Performance

### Observations

- Internal Dice reaches **93.29%**
- Performance remains consistently high across thresholds
- Train, validation, and test metrics are tightly aligned

### Interpretation

Under Frame-Mix:

- The model is exposed to **highly correlated samples across splits**
- It learns **fine-grained visual patterns specific to the dataset**
- It achieves near-perfect segmentation on internal evaluation

### Critical Point

> The model appears highly accurate, but this performance is confined to the biased data distribution.

---

## 3. Patient-Aware Behavior: Constrained Generalization

### Observations

- Internal Test Dice: **67.70%**
- Independent Holdout Dice: **69.59%**
- Optimal threshold shifts to **low confidence (0.1)**

### Interpretation

Under Patient-Aware:

- The model is evaluated on **fully unseen patients**
- It cannot rely on memorized structures
- Predictions become **less confident but more realistic**

### Generalization Stability

> Internal test performance closely matches independent holdout performance.

---

## 4. Recall Limitation on Independent Data

Across both protocols, evaluation on independent patients reveals:

- **Precision:** ~90%+
- **Recall:** ~56–58%

### Interpretation

- The model is **conservative**
- It produces reliable predictions when it predicts
- It **misses a significant portion of pathological regions**

### Clinical Meaning

> The model captures the most obvious parts of lesions but fails to segment their full extent.

---

## 5. Generalization Dynamics Across Epochs (Frame-Mix)

| Epoch | Internal Test Dice | Holdout Mean Dice |
|------|--------------------|-------------------|
| 30 | 92.45% | 73.66% |
| 40 | 93.07% | 72.24% |
| 50 | 93.29% | 70.17% |

### Observation

- Internal performance improves with training
- Independent performance **decreases after Epoch 30**

### Interpretation

- The model progressively specializes to the biased distribution
- Best generalization occurs **before full convergence**
- Later epochs reinforce **dataset-specific memorization**

---

## 6. Structural Effect of the Protocol

### Frame-Mix

- Encourages **memorization of visual redundancy**
- Produces **overconfident predictions**
- Inflates internal metrics
- Leads to **large generalization gaps**

### Patient-Aware

- Enforces **true distribution shift**
- Produces **uncertain but calibrated predictions**
- Aligns internal and external evaluation
- Reflects **real deployment conditions**

---

## 7. Clinical Implications

The discrepancy between protocols has direct consequences:

- A model evaluated at **93% Dice** (Frame-Mix) appears highly reliable
- The same model performs around **70% Dice** on unseen patients
- Recall limitations imply **partial lesion detection**

### Risk

> The model may provide incomplete segmentations while appearing highly confident.

This creates a risk of:

- Underestimating disease extent  
- Misinterpreting model reliability  
- Over-trusting automated outputs  

---

## 8. Final Conclusion

The behavior of the robust TransUNet is strongly dependent on the data partitioning strategy.

- Under Frame-Mix:
  - Performance appears near-perfect
  - Metrics are inflated by data correlation

- Under Patient-Aware:
  - Performance is lower but stable
  - Metrics reflect true generalization

### Final Insight

> The same model can appear clinically reliable or significantly limited depending solely on the evaluation protocol.

The realistic performance of this architecture on unseen patients is consistently observed around:

> **~70% Dice**

Any substantially higher internal score must be interpreted in the context of the data splitting strategy.

---
