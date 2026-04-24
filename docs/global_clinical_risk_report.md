# Global Comparative Analysis: Clinical Risk Induced by Misleading Metrics in Medical AI

This final audit consolidates the results obtained across four experimental configurations:

- **Baseline TransUNet**
- **Robust TransUNet (ResNetV2 + ViT)**
- Evaluated under both:
  - **Frame-Mix (leakage-prone)**
  - **Patient-Aware (clinically valid)**

The objective is not to highlight limitations of deep learning optimization, but to expose a **critical clinical risk**: how flawed data partitioning can generate **artificially inflated metrics**, misleading researchers and endangering patients in real-world deployment.

---

## 1. Ground Truth Matrix: Internal vs Real Performance

The table below summarizes performance at convergence (Epoch 50), comparing:

- **Reported performance** (internal test)
- **True performance** (independent holdout)

| Architecture | Data Split | Reported Dice (Internal Test) | Real Dice (Holdout) | Delta (Δ) |
|--------------|------------|-------------------------------|----------------------|-----------|
| Baseline | Patient-Aware | 67.70% | 69.59% | +1.89% (Reliable) |
| Baseline | Frame-Mix | 86.30% | 69.57% | -16.73% (Misleading) |
| Robust | Patient-Aware | 67.70% | 69.59% | +1.89% (Reliable) |
| Robust | Frame-Mix | 93.29% | 70.17% | -23.12% (Critical Bias) |

### Key Insight

- **Patient-Aware split produces consistent and trustworthy estimates**
- **Frame-Mix artificially inflates performance by up to +23 Dice points**

---

## 2. The Power Paradox: Stronger Models Amplify the Bias

A central finding of this study is the **inverse relationship between model capacity and evaluation reliability under data leakage**.

### Observations

- Under **Patient-Aware**:
  - Both architectures converge to ~69–70% Dice
  - Increased capacity does **not improve generalization**

- Under **Frame-Mix**:
  - Baseline reaches ~86%
  - Robust model reaches ~93%

### Interpretation

The additional capacity of the robust model is not used to learn better representations of pathology. Instead, it is used to:

- Memorize patient-specific anatomical structures  
- Capture acquisition artifacts  
- Exploit temporal and spatial redundancy  

### Conclusion

> Increasing model complexity on biased data does not improve intelligence—it improves the model’s ability to overfit.

---

## 3. Clinical Risk: The Recall Trap

The most critical issue emerges when analyzing **pixel-level metrics on independent patients**.

### Observed Behavior (Robust Model, Holdout)

- **Precision:** >90%  
- **Recall:** ~57–58%

### Interpretation

- The model is highly **conservative**
- Predictions are **accurate when present**
- However, a large portion of lesions remains undetected

### Clinical Consequence

> Up to **40% of pathological regions are missed** on unseen patients.

---

### Failure Scenario

A model validated under Frame-Mix reports:

> **93.29% Dice**

A clinician using this system assumes near-perfect reliability.

However, in real conditions:

- The model detects only the **core of the lesion**
- It fails to capture **diffuse inflammatory margins**
- The segmentation appears precise but is **incomplete**

### Risk

This leads to:

- Underestimation of disease severity  
- Inaccurate clinical assessment  
- Potentially inappropriate treatment decisions  

---

## 4. Audit Verdict

This study demonstrates that the pursuit of high performance in medical AI is fundamentally flawed when evaluation protocols are not rigorously controlled.

### Key Findings

- **Performance Ceiling**
  - True segmentation capability of TransUNet (without domain-specific pretraining or heavy augmentation):
  
  > **~70% Dice on unseen patients**

- **Impact of Data Leakage**
  - Artificial gain of **+15 to +23 Dice points**
  - Entirely driven by **temporal and anatomical correlation**

- **Protocol Integrity**
  - Patient-Aware split is the **only clinically valid methodology**
  - It provides realistic, reproducible, and deployable metrics

---

## 5. Scientific and Clinical Implications

### For Research

- Reported SOTA results must be interpreted with caution
- High Dice scores (>90%) in sequential medical imaging are highly suspicious without:
  - Strict patient-level isolation  
  - Independent external validation  

### For Clinical Deployment

- Models validated under biased protocols cannot be trusted
- Apparent performance does not reflect real-world behavior
- Deployment of such systems introduces **silent clinical failure modes**

---

## 6. Final Statement

> Any medical AI model evaluated without strict patient-level separation is at risk of producing misleading metrics.

> These metrics do not reflect true diagnostic capability, but rather the model’s ability to exploit data leakage.

### Final Insight

- The issue is not model architecture  
- The issue is **evaluation methodology**

And when the methodology is flawed:

> **The model does not become better—it becomes dangerously convincing.**

---

---

## Postscript: On Overfitting and Dataset Limitations

All four configurations exhibit varying degrees of **overfitting**, as evidenced by:

- High training performance relative to validation (especially in Patient-Aware)
- Performance degradation at later epochs on independent data
- Sensitivity to threshold selection in strict settings

This behavior is expected given:

- The **limited dataset size**
- The absence of **advanced data augmentation strategies**
- The use of **random initialization** without domain-specific pretraining

### Important Clarification

These limitations **do not invalidate the conclusions of this study**.

The objective of this audit is not to maximize absolute performance, but to:

- Evaluate the **impact of data partitioning strategies**
- Measure the **reliability of reported metrics**

### Key Point

> Overfitting affects all models equally under both protocols.

However:

- Under **Frame-Mix**, overfitting is **hidden and amplified**, leading to misleadingly high metrics  
- Under **Patient-Aware**, overfitting is **visible and measurable**, allowing proper interpretation  

### Final Note

Improving the dataset (more patients, better augmentation, domain pretraining) would likely:

- Increase absolute performance  
- Reduce overfitting  

But it would **not change the core finding**:

> **Data leakage fundamentally corrupts evaluation, regardless of model quality.**

---
