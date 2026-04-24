# Problem Statement: Reliability of Evaluation in Medical Imaging

---

## 1. The Performance Paradox in Medical AI

Recent scientific literature reports that many segmentation models based on Transformer architectures (e.g., TransUNet) achieve Dice scores exceeding **85%, sometimes even 90%**.

However, when these models are deployed in real clinical settings (i.e., on unseen patients), practitioners frequently observe a **significant drop in performance**, making the tool unreliable for actual diagnosis.

This project is based on a key observation:

> A high Dice score is not necessarily evidence of clinical competence, but may instead reflect a methodological bias known as **data leakage**.

---

## 2. Ultrasound Physics vs Classical Computer Vision

The root of the problem lies in a fundamental misunderstanding between **natural image datasets** and **sequential medical imaging**.

### Classical Computer Vision (e.g., ImageNet)

- Each image is **independent**
- A dataset of 1,000 images corresponds to 1,000 different contexts
- A random split is **statistically valid**

### 2D Ultrasound Imaging

- Images are acquired as a **video stream**
- Typical acquisition rate: **15–30 frames per second**
- Two consecutive frames of the same anatomical structure are **~99% visually identical**

---

## 3. Data Leakage: Unintentional Cheating

When dataset splitting is performed at the **frame level** instead of the **patient level**, the following issues arise:

### Test Set Contamination

- The test set contains images that are **immediate neighbors** of training images

### Spatial Memorization

- The model does **not learn clinical features** of Crohn’s disease
- Instead, it memorizes:
  - Acoustic signature of the sequence
  - Image texture (speckle pattern)
  - Gain settings
  - Patient-specific anatomy

### Outcome

- The model performs **interpolation rather than generalization**
- It effectively “recognizes” images it has already seen, with minimal variation

---

## 4. Key Audit Challenges

This audit aims to quantify this bias by isolating two critical dimensions:

| Challenge               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Generalization         | Can the model segment a patient with completely unseen anatomy and acoustic properties? |
| Robustness             | Does the Dice score remain stable when moving from internal test to an independent holdout? |
| Scientific Integrity   | How can we define a split protocol that guarantees the absence of temporal correlation? |

---

## 5. Final Objective

The goal is to demonstrate that **clinical validity requires patient-aware evaluation protocols**.

### Hypotheses

- A model trained with the **Frame-Mix protocol (biased)** will:
  - Show artificially high performance on its internal test set  
  - Fail on independent patients  

- A model trained with a **Patient-Aware protocol (robust)** will:
  - Show lower apparent performance  
  - Maintain **consistent and reliable results** across diverse clinical cases  

---
