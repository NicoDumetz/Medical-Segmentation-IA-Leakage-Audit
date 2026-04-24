# Metrics and Evaluation Protocol

This document defines the performance indicators and the stress-testing methodology used to assess the reliability and scientific integrity of segmentation models.

---

## 1. Reference Metrics for the Audit

To avoid misleading conclusions, we rely on overlap-based and clinically relevant metrics, focusing on lesion detection rather than background dominance.

### Dice Coefficient (F1-Score) — Primary Metric

The Dice coefficient measures similarity between prediction (P) and ground truth (GT), giving double weight to the intersection.

```text
Dice = (2 × |P ∩ GT|) / (|P| + |GT|)
```

**Role:**

- Evaluates global agreement
- Balances false positives and false negatives

### Intersection over Union (IoU) — Strict Metric

Also known as the Jaccard Index, IoU is more demanding than Dice.

```text
IoU = |P ∩ GT| / |P ∪ GT|
```

**Role:**

- Measures contour precision
- Reflects segmentation accuracy at the boundary level

### Precision — Reliability Metric

Precision measures how many predicted lesion pixels are actually pathological.

```text
Precision = TP / (TP + FP)
```

**Role:**

- Reduces false positives
- Answers: when the model predicts a lesion, is it correct?

### Recall — Sensitivity Metric

Recall measures how much of the real lesion area was detected.

```text
Recall = TP / (TP + FN)
```

**Role:**

- Reduces false negatives
- Answers: did the model detect the full inflammatory region?

---

## 2. Why Accuracy is Excluded and Other Misleading Metrics

### A. Class Imbalance Issue

In gastrointestinal ultrasound:

- The lesion (foreground) represents only 2% to 5% of the image
- The remaining 95% to 98% corresponds to background

### B. The "Lazy Predictor" Problem

A model that predicts no lesion at all (all pixels = background) would achieve:

- ~98% Accuracy on a typical patient

**Implication:**

- Scientific interpretation: High performance
- Clinical reality: Complete failure

### C. Additional Misleading Metrics

#### Accuracy (Global Pixel Accuracy)

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- Dominated by background pixels
- Rewards trivial predictions

#### Specificity (True Negative Rate)

```text
Specificity = TN / (TN + FP)
```

- Inflated due to dominant background
- Not informative for lesion detection

#### ROC-AUC (Pixel-wise)

```text
AUC = ∫ TPR(FPR) dFPR
```

- Can remain high despite poor segmentation
- Threshold-independent, masking real decision behavior

#### Mean Pixel Accuracy

```text
Mean Accuracy = (1/N) Σ_i (TP_i / (TP_i + FN_i))
```

- Same limitation as global accuracy
- Not adapted to highly imbalanced segmentation

---

## 3. Audit Protocol: Validation Cascade

Data leakage cannot be identified through a single metric. It is revealed through performance degradation across increasingly strict evaluation stages.

### Level 1: Convergence Monitoring (Validation Set)

- When: During training (train.py)

- Fraud indicator:
  If the validation Dice curve closely follows the training curve from the beginning, the model is memorizing rather than generalizing.

### Level 2: Threshold-Based Evaluation (Internal Test Set)

- When: Post-training (threshold_test.py)

- Method:
  Evaluation across 9 confidence thresholds (0.1 to 0.9)

- Fraud indicator:
  Artificially high and stable scores (e.g., >85%) indicate prior exposure to test samples.

### Level 3: Final Verdict (Independent Holdout)

- When: Final evaluation (independent.py)

- Data:
  Strictly isolated patients: CD CaeC, P53, P56A, P58A

- Interpretation:
  This is the only reliable estimate of clinical performance

- Key signal:
  The drop in Dice between internal test and independent holdout directly measures data leakage

---
