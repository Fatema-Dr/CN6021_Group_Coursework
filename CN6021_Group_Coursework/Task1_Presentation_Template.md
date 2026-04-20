# Task 1 Presentation Template: Customer Churn Prediction

This template is designed to help you secure the maximum **10 marks** for the CN6021 Task 1 group presentation by focusing on technical justification, visual evidence, and business impact.

---

## Slide 1: Title & Team
*   **Title:** Customer Churn Prediction using a Custom Shallow Neural Network
*   **Subtitle:** CN6021: Advance Topic in AI and Data-Science
*   **Content:** 
    *   Team Member Names
    *   Student IDs
    *   Date of Presentation

## Slide 2: Problem Definition & Dataset
*   **Objective:** Develop an automated system to predict customer churn for a subscription service.
*   **Dataset Overview:** Briefly describe the dataset used (e.g., Telco, Bank, or KKBox).
*   **The Challenges:** 
    *   High Dimensionality (redundant features)
    *   Non-Linear Relationships
    *   Significant Class Imbalance (more retained than churned)

## Slide 3: Exploratory Data Analysis (EDA)
*   **Visuals:** Target Distribution Pie/Bar Chart (showing the imbalance).
*   **Key Insight:** "Our dataset shows a 20/80 churn-to-retention ratio, which heavily biases a standard model."
*   **Correlation Heatmap:** Identify the top 3-5 features most correlated with churn.

## Slide 4: Addressing Data Challenges
*   **Feature Selection:** Mention dimensionality reduction (e.g., Mutual Information, PCA, or RFE).
*   **Handling Imbalance:** Explain the technique used (e.g., SMOTE oversampling, undersampling, or weighted loss).
*   **Preprocessing:** List scaling methods (StandardScaler/MinMaxScaler) and categorical encoding.

## Slide 5: Custom Neural Network Architecture (The "NumPy" implementation)
*   **Architecture Diagram:** Simple visual (Input Layers -> Hidden Layer -> Output).
*   **Implementation:** Explicitly state: "Built entirely using NumPy for deep mathematical control."
*   **Justification:** 
    *   **Activation:** ReLU for hidden (avoids vanishing gradients), Sigmoid for output (binary probability).
    *   **Units:** Why you chose X hidden units (e.g., "32 units to capture complexity without overfitting").

## Slide 6: Training & Optimization
*   **Visuals:** Loss and Accuracy curves (Training vs. Validation).
*   **Hyperparameters:** A small table showing:
    *   Learning Rate (e.g., 0.01)
    *   Epochs (e.g., 100)
    *   Batch Size (e.g., 32)
*   **Optimizer:** Mention SGD, Adam, or Momentum implementation.

## Slide 7: Performance Evaluation
*   **Visuals:** Confusion Matrix and ROC-AUC Curve.
*   **Key Metrics Table:** Precision, Recall, F1-Score, and AUC-ROC.
*   **Analysis:** "We prioritized **Recall** because the business cost of missing a churner is higher than the cost of a false alarm."

## Slide 8: Interpretability Analysis (High-Mark Section)
*   **Visual:** Feature Importance Bar Chart.
*   **Explanation:** "Our model identified 'Contract Type' and 'Monthly Charges' as the primary churn drivers."
*   **Insight:** Explain how one specific feature influences the prediction (e.g., "Longer contracts significantly reduce churn probability").

## Slide 9: Conclusion & Business Impact
*   **Summary:** How the model solves the stakeholders' needs.
*   **Actionable Recommendation:** "Stakeholders should target high-charge, month-to-month customers with 12-month loyalty discounts to reduce churn by X%."

## Slide 10: Q&A
*   "Thank you for your time. We welcome any questions regarding our implementation or results."

---

## Top 3 Tips for a High Mark (10/10)
1.  **Explain the "Why":** Don't just say *what* you did; explain *why* it was the best choice for the constraints (e.g., "We used a shallow network because computational resources were limited").
2.  **Speak, Don't Read:** Use the slides for visual proof (graphs/charts) and use your voice to tell the story of the data.
3.  **Demonstrate the NumPy Logic:** If asked, be ready to explain how you implemented the gradient descent or backpropagation manually without libraries like PyTorch.
