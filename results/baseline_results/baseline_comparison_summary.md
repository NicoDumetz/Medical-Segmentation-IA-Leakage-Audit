# Comparative Analysis: Frame-Mix Illusion and Clinical Risk

This report compares two training paradigms applied to the same architecture: **Baseline TransUNet**.

The only variable changed is the **data splitting strategy**.

The results demonstrate that a methodological error, namely frame-level mixing, can create an artificial performance gain and lead to misleading scientific conclusions.

---

## 1. The Misleading Performance: Frame-Mix Split

In the Frame-Mix protocol, images from the same patient/sequence are distributed across training, validation, and test sets.

### Reported Performance

| Metric | Value |
|--------|-------|
| Train Dice (Epoch 50) | 82.28% |
| Validation Dice (Epoch 50) | 86.29% |
| Internal Test Dice (Threshold 0.5) | 86.30% |

### Anomaly Analysis

The model appears highly performant. However, the validation score is higher than the training score, which is a strong indicator of **data leakage**.

The model did not learn robust biomarkers of Crohn’s disease. Instead, it likely memorized patient-specific visual signatures, including:

- Ultrasound texture
- Probe angle
- Acquisition noise
- Static anatomical patterns

This behavior corresponds to **spatial interpolation**, not clinical generalization.

---

## 2. The Reality Check: Patient-Aware Split

In the Patient-Aware protocol, the dataset is strictly separated at the patient/sequence level. A patient seen during training never appears in validation or test.

### Reported Performance

| Metric | Value |
|--------|-------|
| Train Dice (Epoch 50) | 82.14% |
| Validation Dice (Epoch 50) | 48.14% |
| Internal Test Dice (Threshold 0.1) | 62.58% |

### Interpretation

Once the model is evaluated on unseen patients, performance drops substantially.

The Train Dice remains high, showing that the architecture has sufficient capacity to fit the training distribution. However, the Validation Dice collapse reveals classical overfitting:

- The model learns training-set textures
- The optimizer fits patient-specific patterns
- Generalization to unseen anatomy remains limited

The transition from **86.30%** to **62.58%** exposes the true performance range of the baseline pipeline.

---

## 3. Clinical Crash-Test: Independent Holdout

Both models were evaluated on a fully independent holdout set composed of unseen patients.

| Model | Mean Image Dice (Holdout) | Delta vs Internal Test |
|-------|---------------------------|-------------------------|
| Baseline Frame-Mix | 69.57% | -16.73% |
| Baseline Patient-Aware | 66.52% | +3.94% |

### Key Finding

The Frame-Mix model collapses from **86.30%** to **69.57%** when evaluated outside its biased internal split.

Both models converge toward a similar real-world performance range: **66–69% Dice**.

This confirms that the additional performance observed in the Frame-Mix protocol exists only inside the contaminated evaluation setup.

---

## 4. Clinical Risk of Data Leakage

This comparison highlights three critical points for medical AI evaluation.

### 1. The Illusion of High Dice Scores

Very high scores, especially above **85–90%**, in sequential ultrasound segmentation should be treated with caution.

They may indicate:

- Frame-level leakage
- Patient overlap
- Temporal correlation between training and test samples

### 2. Artificial Dice Inflation

On this cohort, frame mixing produces an artificial gain of nearly **20 Dice points** compared with strict patient-aware testing.

### 3. Clinical Deployment Risk

A Frame-Mix model may appear reliable during evaluation while failing on new patients.

This creates a major clinical risk:

- The clinician expects a model performing around **86% Dice**
- In reality, the model operates closer to **69% Dice**
- Recall degradation means that pathological regions may be missed

---

## 5. Final Verdict

The real performance of the Baseline TransUNet architecture, trained with a naive pipeline, is between:

> **62% and 69% Dice**

This setup includes:

- Destructive resizing
- Limited augmentation
- No pretrained weights
- Basic SGD optimization
- Random initialization

Any substantially higher result under the same conditions should be considered a likely artifact of evaluation bias.

### Final Conclusion

The Frame-Mix protocol does not measure clinical competence.

It measures the model’s ability to exploit temporal redundancy.

For medical segmentation, only a strict **Patient-Aware evaluation protocol** provides a scientifically valid estimate of generalization.

---
