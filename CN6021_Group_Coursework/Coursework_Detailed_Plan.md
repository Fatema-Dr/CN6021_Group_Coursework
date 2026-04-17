# CN6021: Advance Topic in AI and Data-Science - Coursework Summary

## Overview
*   **Module Code:** CN6021
*   **Assignment Weighting:** 75%
*   **Submission Date:** 23:59, 01 May 2026
*   **Word Count:** 2,000 words maximum (excluding Appendix)
*   **Format:** Group Work (Consolidated single document)

---

## Task 1: Customer Churn Prediction (40%)

### Objective
Develop a system to predict customer churn for a subscription-based service, addressing challenges like high dimensionality, non-linear relationships, and class imbalance.

### Key Requirements
1.  **Data Challenges:** 
    *   Implement feature selection or dimensionality reduction.
    *   Explain how your network handles non-linear relationships.
    *   Address class imbalance (oversampling, undersampling, or weighted loss).
2.  **Implementation:** 
    *   Build a **Shallow Neural Network using NumPy** (or similar low-level library).
    *   Justify architectural choices (activation functions, hidden units).
3.  **Optimization & Evaluation:** 
    *   Hyperparameter tuning.
    *   Use appropriate metrics: Precision, Recall, F1-score, AUC-ROC.
4.  **Interpretability:** 
    *   Develop a method for feature importance analysis.
    *   Explain how different features influence predictions.

### Recommended Datasets
*   **Telco Customer Churn (Kaggle/IBM):** The standard benchmark. Features include demographics, account info, and services. Good for demonstrating feature engineering.
*   **Bank Customer Churn (Kaggle):** 10,000 records with financial features (Credit Score, Balance). Excellent for testing non-linear relationships.
*   **KKBox’s Churn Prediction (Kaggle):** Real-world subscription data with transaction logs. Best for handling massive, imbalanced datasets.

---

## Task 2: 3D Semantic Segmentation for Brain Tumours (35%)

### Objective
Develop an automated system to detect and segment brain tumours from 3D MRI scans.

### Key Requirements
1.  **Implementation:**
    *   Design a **3D Convolutional Neural Network (CNN)** using **PyTorch**.
    *   Justify architectural choices (layers, activation, skip connections).
2.  **Pipeline:**
    *   Custom data loading using `torch.utils.data.Dataset` and `DataLoader`.
    *   3D data augmentation (rotations, flips, elastic deformations).
3.  **Advanced Techniques:**
    *   Address class imbalance using Dice loss, Focal loss, or weighted cross-entropy.
    *   Explore **Transfer Learning** using pretrained 2D or 3D CNN models.
4.  **Optimization & Evaluation:**
    *   Custom training loop with GPU acceleration.
    *   Metrics: Dice score, IoU, Hausdorff distance.

### Recommended Datasets
*   **BraTS 2024 (Synapse/TCIA):** The gold standard for brain tumor segmentation. Provides four 3D MRI modalities (T1, T1Gd, T2, T2-FLAIR) with expert labels.
*   **Medical Decathlon (Task01_BrainTumour):** A cleaned subset of BraTS, often easier for initial pipeline development.
*   **The Cancer Imaging Archive (TCIA):** Hosts historical BraTS datasets (2012–2023) and the BraTS-Africa collection (good for showing model generalizability).

---

## Assessment Criteria Summary

| Criteria | Task 1 (40 Marks) | Task 2 (30 Marks) |
| :--- | :--- | :--- |
| **Model Functionality** | Shallow NN (6) | 3D PyTorch CNN (7.5) |
| **Data Handling** | Dimensionality/Imbalance (6) | 3D Augmentation/Pipeline (7.5) |
| **Optimization/Metrics** | Tuning & Eval (6) | Tuning & GPU Training (7.5) |
| **Analysis/Reporting** | Interpretability & Logic (6) | Clarity & Results (7.5) |
| **Visual Documentation** | Screenshots & Methods (6) | N/A |
| **Presentation** | Week 11 Group Presentation (10) | N/A |

---

## Deliverables
1.  **Python Code:** Clean, commented implementation for both tasks.
2.  **Report:** Max 2,000 words covering introduction, methodology, results, and conclusions.
3.  **Presentation:** Summarizing approach and insights for Task 1 (Week 11).
