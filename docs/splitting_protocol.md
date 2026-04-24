# Technical Specifications of the Partitioning Protocol

This document outlines the structural characteristics of the two partitioning protocols used in the `Dataset_Comparison` bundle.  
The unit of analysis for this audit is the **patient/sequence group**.

---

## 1. Source Dataset Inventory

The dataset consists of manually segmented ultrasound images.

- **Total volume:** 17,488 images  
- **Composition:**  
  - 11,396 TP (pathological)  
  - 6,092 TNP (non-pathological)  
- **Entities:** 50 distinct patient/sequence groups  

---

## 2. Definition of Experimental Protocols

Two data distribution strategies were applied on a common subset of **13,256 images (46 groups)**, after isolating an independent holdout set.

---

### A. "Dataset" Protocol (Group-Based Split)

This protocol distributes data at the **patient/sequence group level**.

- **Method:**  
  Each patient/sequence group is assigned exclusively to either Train, Validation, or Test.

- **Data Leakage:**  
  No image from a given group appears in more than one set (**strict separation**).

- **Distribution:**  
  - Train: 34 groups  
  - Validation: 4 groups  
  - Test: 8 groups  

---

### B. "Frame-Mix" Protocol (Image-Based Split)

This protocol distributes data at the **individual image level**, regardless of group origin.

- **Method:**  
  Random shuffle of the 13,256 images followed by an 80/10/10 split.

- **Data Leakage:**  
  Images from the same patient/sequence group can appear across Train, Validation, and Test sets (**full overlap**).

- **Distribution:**  
  - Train: 46 groups represented  
  - Validation: 45 groups represented  
  - Test: 45 groups represented  

---

## 3. Independent Holdout Set (Common Reference)

A subset of **4 patient groups** was extracted prior to any splitting operation to serve as an external evaluation set.  
This holdout is identical for both protocols.

- **Identifiers:**  
  CD CaeC, P53, P56A, P58A  

- **Volume:**  
  - 4,232 images  
  - 2,882 TP / 1,350 TNP  

- **Constraint:**  
  - 0% overlap with Train and Validation sets in both protocols  

---

## 4. Summary Table

| Parameter                | Dataset Protocol             | Frame-Mix Protocol           |
|------------------------|-----------------------------|------------------------------|
| Split Logic            | Patient/sequence group      | Individual image             |
| Train Entities         | 34 groups                   | 46 groups                    |
| Validation Entities    | 4 groups                    | 45 groups                    |
| Test Entities          | 8 groups                    | 45 groups                    |
| Train/Test Overlap     | 0 groups                    | 45 groups                    |

---

## 5. Build Details (`build_comparison_datasets.py`)

The dataset bundle is generated through the following steps:

1. Extraction of full identifiers  
   (including sequence numbers to distinguish all 50 entities)

2. Random selection of 4 holdout groups  
   (`seed = 42`)

3. Creation of the **Dataset protocol**  
   using strict group-based splitting

4. Creation of the **Frame-Mix protocol**  
   via random image-level shuffling

5. Strict alignment of image/mask pairs  
   for both protocols
