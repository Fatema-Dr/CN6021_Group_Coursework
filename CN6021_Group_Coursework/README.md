# Task 1: Customer Churn Prediction Using a Shallow Neural Network

## Overview

This project implements a complete machine learning pipeline for predicting customer churn using a **shallow neural network built from scratch using NumPy**. The implementation demonstrates a thorough understanding of forward propagation, backpropagation, and gradient descent from first principles.

**Course:** CN6021 — Advanced Topics in AI and Data Science

## Dataset

The project uses the [Ecommerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) from Kaggle, containing:
- **50,000 customer records**
- **25 features** spanning demographics, platform engagement, purchase behaviour, customer service interactions, and financial indicators
- **Target variable:** `Churned` (binary: 1 = churned, 0 = retained)
- **Class imbalance:** ~71.1% retained vs ~28.9% churned (2.46:1 ratio)

## Approach

### Pipeline Steps
1. **Data Loading & EDA** — Target distribution, correlation analysis, feature distributions, PCA visualization
2. **Preprocessing** — Outlier handling, missing value imputation, one-hot encoding, stratified train/val/test split, feature scaling
3. **Feature Selection** — Correlation-based removal (|r| > 0.90) + Mutual Information filtering (MI > 0.001)
4. **Class Imbalance** — Weighted Binary Cross-Entropy loss with class weights
5. **Shallow Neural Network** — Single hidden layer with ReLU activation, pure NumPy implementation
6. **Hyperparameter Grid Search** — 27 configurations evaluated on validation set
7. **Final Model Training** — With early stopping based on validation loss
8. **Evaluation** — Classification report, AUC-ROC, confusion matrix, ROC curve
9. **Threshold Optimization** — Precision-Recall analysis and F1-optimized threshold selection
10. **Interpretability** — Weight-based and permutation feature importance, churn risk profiles

### Neural Network Architecture
```
Input (n features) → Hidden Layer (32-128 units, ReLU) → Output (1 unit, Sigmoid)
```

Key design choices:
- **ReLU activation** — Captures non-linear relationships, avoids vanishing gradients
- **He initialization** — Maintains variance for ReLU activations
- **Weighted BCE loss** — Addresses class imbalance at the objective level
- **L2 regularization** — Prevents overfitting on high-dimensional inputs
- **Mini-batch SGD** — Balances convergence speed and gradient stability
- **Early stopping** — Halts training when validation loss plateaus

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Additional Requirements (for Quarto report)

If you want to generate the PDF report from `Task1.qmd`:

1. Install [Quarto](https://quarto.org/docs/getting-started/installation.html)
2. Install Jupyter: `pip install jupyter`

## Usage

### Running the Pipeline

```bash
# Run the full pipeline (generates all outputs)
python task1_churn_prediction\ \(1\).py
```

**Note:** On Windows, you may need to escape the parentheses in the filename or rename the file to remove them.

### Configuration

Edit the configuration section at the top of `task1_churn_prediction (1).py`:

```python
RANDOM_STATE = 42
SHOW_PLOTS = False  # Set True to display figures interactively
OUTPUT_DIR = "outputs"
DATA_PATH = "data/ecommerce_customer_churn_dataset.csv"
```

### Output Files

The pipeline generates the following output files in the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_target_distribution.png` | Target variable bar chart |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_feature_distributions.png` | Histograms by churn status |
| `04_pca_scatter.png` | PCA 2D projection (EDA only) |
| `05_mutual_information.png` | MI feature ranking |
| `06_training_curves.png` | Loss and accuracy curves |
| `07_confusion_matrix.png` | Confusion matrix heatmap |
| `08_roc_curve.png` | ROC curve |
| `09_pr_curve_threshold.png` | Precision-recall curve and F1 vs threshold |
| `10_feature_importance.png` | Weight-based and permutation importance |
| `churn_model.pkl` | Trained model, scaler, and configuration |

## Key Results

Based on the implementation in `Task1.qmd`:

- **AUC-ROC:** ~0.92
- **Macro F1:** ~0.86
- **Churned Class Recall:** ~85%
- **Optimal Threshold:** ~0.35 (vs default 0.5)

### Top Predictive Features

1. **Lifetime_Value** — Single most influential feature
2. **Customer_Service_Calls** — Strong dissatisfaction indicator
3. **Cart_Abandonment_Rate** — Early warning behavioural indicator
4. **Days_Since_Last_Purchase** — Recency indicator (RFM framework)

## Project Structure

```
CN6021_Group_Coursework/
├── task1_churn_prediction (1).py   # Main Python pipeline
├── Task1.qmd                        # Quarto document for report
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── references.bib                   # Bibliography
├── data/
│   └── ecommerce_customer_churn_dataset.csv
└── outputs/
    ├── 01_target_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_distributions.png
    ├── 04_pca_scatter.png
    ├── 05_mutual_information.png
    ├── 06_training_curves.png
    ├── 07_confusion_matrix.png
    ├── 08_roc_curve.png
    ├── 09_pr_curve_threshold.png
    ├── 10_feature_importance.png
    └── churn_model.pkl
```

## License

This project is for educational purposes as part of CN6021 — Advanced Topics in AI and Data Science.