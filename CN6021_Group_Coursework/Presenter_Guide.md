# Presenter Script & Technical Guide
## Project: Customer Churn Prediction (CN6021)
### Team Members & Role Assignments:
1. **Shyam Vijay Jagani (2611208)**: Introduction & Exploratory Data Analysis (Slides 1-7)
2. **Jasmi Alasapuri (2571395)**: Methodology & Technical Architecture (Slides 8-11)
3. **Fatema Doctor (2604383)**: Training, Results & Final Performance (Slides 12-16)
4. **Parth Rathwa (2509367)**: Business Impact, Action Plan & Conclusion (Slides 17-22)

---

## 🖼️ Slide-by-Slide Breakdown

### 👤 Part 1: Problem Definition (Presenter: Shyam Vijay Jagani)
- **Slide 1 (Title):** "Welcome everyone. Today we are presenting our project on Customer Churn Prediction. We developed a custom shallow neural network from scratch to identify which customers are most likely to leave, allowing for proactive retention strategies."
- **Slide 2 (# Problem Definition & Dataset):** "In this first section, we'll define the business problem and look at the characteristics of the data we used to train our AI."
- **Slide 3 (## Project Objective):** "Our goal was to build an automated system for an e-commerce subscription service. We dealt with 50,000 customers and 25 behavioral features. A critical challenge was the 28.9% churn rate, meaning our data is 'imbalanced'—most customers stay, which makes finding the ones who leave like finding a needle in a haystack."

  **Technical Keywords:**
  - **Churn Rate:** The percentage of customers who stop using a service over a given time. Our ~29% is significant for an e-commerce company.
  - **Imbalanced Class:** When one category in your data (churners) is much smaller than the other (retained). This makes it harder for the AI to learn what a "churner" looks like.
  - **Non-linear Relations:** Complex patterns where the outcome isn't just a simple straight-line relationship (e.g., age alone doesn't predict churn, but age *combined* with purchase history might).
  - **High Dimensionality:** Having a large number of features (25+). This provides more detail but can also confuse the model with "irrelevant noise."

### 👤 Part 2: EDA (Presenter: Shyam Vijay Jagani)
- **Slide 4 (# Exploratory Data Analysis):** "Before building the model, we performed EDA to find the 'signal in the noise' and discover what actually drives customer behavior."
- **Slide 5 (## Target Distribution):** "Almost 71% of our data is retained. If a model just guessed 'No Churn' every time, it would be 71% accurate but totally useless for the business. That's why we prioritize AUC-ROC and Weighted Loss to ensure we capture the minority churn class."

  **Technical Keywords:**
  - **Accuracy:** The percentage of correct guesses. In imbalanced data, accuracy is "misleading" because you can be 70% accurate by never predicting a single churner.
  - **AUC-ROC:** A combined score (0 to 1) that measures how good the model is at separating the two classes (Staying vs. Leaving) across all possible confidence levels.
  - **Weighted Loss:** We tell the AI that missing a "churner" is 3x more expensive than missing a "retained" customer. This forces it to focus on the rare class.

- **Slide 6 (## Correlation Insights):** "We looked at what drives churn. Positive correlations (red) mean as value goes up, churn risk goes up—like service calls. Negative correlations (blue) like 'Lifetime Value' act as protective factors."

  **Technical Keywords:**
  - **Correlation:** A value from -1 to +1 showing how two things move together.
  - **Positive Correlation (+):** As X goes up, Y goes up. More service calls = more churn risk.
  - **Negative Correlation (-):** As X goes up, Y goes down. Higher loyalty/years = lower churn risk.

- **Slide 7 (## Behavior & PCA):** "On the left are feature distributions. On the right is a PCA projection. The overlapping points show that the data is non-linearly separable, which justifies the use of a Neural Network over simpler models."

  **Technical Keywords:**
  - **PCA (Principal Component Analysis):** A technique to squash 25 features down into 2D (PC1 and PC2) so we can "see" the data structure. It shows how distinct or overlapping our classes are.
  - **Feature Distributions:** Histograms showing how values (like Age or Credit) spread across churners vs. retained customers.

### 👤 Part 3: Methodology (Presenter: Jasmi Alasapuri)
- **Slide 8 (# Methodology & Engineering):** "This section covers how we cleaned the data and the math behind our custom-built neural network."
- **Slide 9 (## Feature Selection Pipeline):** "We handled outliers, imputed missing values, and scaled the data. Crucially, we used Mutual Information to select the top 25 features that have the strongest relationship with churn."

  **Technical Keywords:**
  - **Winsorization (Outlier Handling):** We "cap" extreme values (like Age=150) to a reasonable maximum (Age=100) so they don't distort the model's math.
  - **Mode/Median Imputation:** Filling missing data gaps. We use the **Median** for numbers and **Mode** (most frequent) for categories.
  - **StandardScaler:** Re-sizes all data (like Age 0-100 and Salary 0-100k) to a standard scale (centered at 0). This prevents large numbers from dominating the model.
  - **Mutual Information:** A score measuring how much a feature "tells us" about the target. Unlike correlation, it finds complex, non-linear relationships.

- **Slide 10 (## Architecture: Custom NumPy SNN):** "This is a key technical achievement. Instead of using pre-made tools like TensorFlow, we built this Neural Network entirely in NumPy. This gives us raw control over the gradient tracking and custom loss functions."

  **Technical Keywords:**
  - **SNN (Shallow Neural Network):** A neural network with just one hidden layer. Perfect for tabular data where "Deep" networks might over-complicate and overfit.
  - **ReLU:** The "Rectified Linear Unit" turns off negative signals. It's the industry-standard "engine" for intermediate layers.
  - **Sigmoid:** A final gate that squashes any output into a probability between 0 and 1 (0% to 100% churn risk).
  - **Chain Rule:** The calculus used to calculate exactly how to "adjust the knobs" (weights) inside the network to reduce error.

- **Slide 11 (## Architecture: Design Decisions):** "We chose ReLU for the hidden layer to prevent vanishing gradients, Sigmoid for the output to get a clean probability, and L2 Regularization to stop the model from overfitting on noise."

  **Technical Keywords:**
  - **L2 Regularization (Lambda):** A penalty we add for being too complex. It forces the model to keep its "weights" small and simple, preventing it from memorizing specific customers.
  - **Vanishing Gradients:** A common math problem where the signals get so small during training that the model stops learning. ReLU fixes this.
  - **Early Stopping (Patience):** We set the model to stop training automatically if it stops improving on new data. This prevents it from "overloading."
  - **Epoch:** One complete pass of the entire dataset through the neural network.

### 👤 Part 4: Training & Results (Presenter: Fatema Doctor)
- **Slide 12 (# Training & Results):** "Now we look at the results of our hyperparameter tuning and how the model learned over time."
- **Slide 13 (## Optimization Strategy):** "We ran 27 different configurations of learning rates and layer sizes. We found that a learning rate of 0.05 hit the highest F1-Score, giving us the most balanced performance."

  **Technical Keywords:**
  - **Grid Search (Hyperparameter Search):** Systematically testing every combination of settings (LR, Layer Size, Lambda) to find the best configuration for the specific data.
  - **Learning Rate (LR):** How big of a "step" the model takes when adjusting its math. Too large = skips the solution; too small = takes forever.
  - **F1-Score:** The "sweet spot" metric. It balances the need to catch churners (Recall) while not being wrong (Precision).

- **Slide 14 (## Training Stability):** "Our training curves show steady learning with no divergence. We used 'Early Stopping' at epoch 195 to halt training at the point of maximum generalization."

  **Technical Keywords:**
  - **Convergence:** When the model's error "flattens out," meaning it has learned as much as it can from the current settings.
  - **Validation Loss:** A measure of error on data the model *hasn't* seen. If this goes up while training loss goes down, you're overfitting.
  - **Overfitting:** When a model "memorizes" specific data points (noise) instead of learning the general pattern.
  - **Generalization:** The ability of the model to work correctly on "unseen" customers it wasn't trained on.

### 👤 Part 5: Final Performance (Presenter: Fatema Doctor)
- **Slide 15 (# Final Performance):** "The bottom line: how does the model perform on completely new, unseen customers?"
- **Slide 16 (## Evaluation Metrics):** "We hit an AUC-ROC of 0.91—an excellent score. Our Precision and Recall are balanced at 81%, meaning we successfully minimize both missed churners and unnecessary outreach costs."

  **Technical Keywords:**
  - **Confusion Matrix:** A table showing "Actual" vs. "Predicted" outcomes. It lays out exactly where the model succeeded or failed (False Positives/Negatives).
  - **ROC Curve:** A graph showing the trade-off between sensitivity and specificity. The more "bowed" toward the top-left, the better the model.
  - **Precision:** "Out of everyone we called a churner, how many actually were?"
  - **Recall:** "Out of everyone who actually churned, how many did we successfully find?"

### 👤 Part 6: Business Impact (Presenter: Parth Rathwa)
- **Slide 17 (# Business Impact):** "Finally, let's translate these technical metrics into actionable business strategies."
- **Slide 18 (## Feature Importance):** "Our model identifies Customer Service Calls and Cart Abandonment as the primary indicators of churn. This gives the marketing and support teams clear areas to focus on."

  **Technical Keywords:**
  - **Permutation Importance:** We "mess up" one feature at a time and see how much the accuracy drops. If it drops a lot, that feature was critical.
  - **Weight-based Importance:** Looking at the "strength" of the internal connections (weights) for each feature. Large weights usually mean high influence.

- **Slide 19 (## Retention Action Plan):** "We recommend targeted outreach for anyone with more than 3 calls, and win-back flows for those with high abandonment. This data-driven approach is far more efficient than mass marketing."

### 👤 Part 7: Conclusion (Presenter: Parth Rathwa)
- **Slide 20 (# Conclusion):** "To wrap up our findings and look at the path forward..."
- **Slide 21 (## Summary):** "We delivered a robust, scratch-built Neural Network with balanced metrics and explainable insights that directly support business retention goals."
- **Slide 22 (## Q&A):** "Thank you for your time. Are there any questions regarding the implementation, the math, or the business strategy?"

---

## 📚 Technical Glossary for the Presenter

| Term | Simple Definition for the Audience |
|:---|:---|
| **Neural Network** | A computer system modeled on the human brain that learns patterns from data. |
| **Backpropagation** | The 'learning' process where the model calculates its error and moves backward to fix its internal settings. |
| **Activation Function** | A math gate (ReLU/Sigmoid) that decides if a signal is strong enough to pass to the next layer. |
| **AUC-ROC** | A score from 0 to 1 measuring how well the model separates two classes. 0.9+ is considered excellent. |
| **F1-Score** | A combined measure of Precision and Recall. Essential for datasets where one class is rare. |
| **Overfitting** | When a model memorizes the training data too well and fails to predict new customers correctly. |
| **NumPy** | A high-performance math library. Building in NumPy shows deep technical proficiency. |
