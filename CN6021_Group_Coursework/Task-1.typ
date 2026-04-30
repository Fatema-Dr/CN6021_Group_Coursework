// ═══════════════════════════════════════════════════════════════════════════
// Task 1: Customer Churn Prediction
// CN6021 — Advanced Topics in AI and Data Science
// ═══════════════════════════════════════════════════════════════════════════

// ── Monochrome Design System ───────────────────────────────────────────────
#let c-black      = rgb("#0a0a0a")    // rich black
#let c-charcoal   = rgb("#1f1f1f")    // primary dark
#let c-graphite   = rgb("#2d2d2d")    // secondary dark
#let c-slate      = rgb("#404040")    // mid tone
#let c-steel      = rgb("#6b6b6b")    // body text
#let c-silver     = rgb("#a3a3a3")    // muted
#let c-platinum   = rgb("#d4d4d4")    // borders
#let c-pearl      = rgb("#e8e8e8")    // light bg
#let c-ivory      = rgb("#f5f5f5")    // off-white
#let c-white      = rgb("#ffffff")    // pure white

// ── Typography ─────────────────────────────────────────────────────────────
#set text(font: "New Computer Modern", size: 10.5pt, fill: c-charcoal, ligatures: true)
#set par(justify: true, leading: 0.75em, spacing: 1.15em)
#set heading(numbering: "1.1.")

// ── Page Setup ─────────────────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: (top: 26mm, bottom: 26mm, left: 24mm, right: 24mm),
  header: context {
    if counter(page).get().first() > 2 {
      grid(
        columns: (1fr, 1fr),
        align: (left, right),
        text(8pt, fill: c-silver, font: "New Computer Modern")[CN6021 — Advanced Topics in AI & Data Science],
        text(8pt, fill: c-silver, font: "New Computer Modern")[Task 1: Customer Churn Prediction],
      )
      v(4pt)
      line(length: 100%, stroke: 0.5pt + c-platinum)
    }
  },
  footer: context {
    if counter(page).get().first() > 2 {
      line(length: 100%, stroke: 0.5pt + c-platinum)
      v(4pt)
      grid(
        columns: (1fr, 1fr),
        align: (left, right),
        text(8pt, fill: c-silver)[Group Coursework],
        text(8pt, fill: c-silver)[Page #counter(page).display("1 / 1", both: true)],
      )
    }
  },
)

// ── Heading Styles ─────────────────────────────────────────────────────────
#show heading.where(level: 1): it => {
  v(1.2em)
  block(width: 100%)[
    #rect(
      width: 100%,
      fill: c-charcoal,
      radius: 3pt,
      inset: (x: 16pt, y: 11pt),
    )[
      #text(font: "New Computer Modern", weight: "bold", size: 14pt, fill: c-white)[
        #counter(heading).display() #it.body
      ]
    ]
  ]
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(1em)
  block[
    #box(
      width: 100%,
      height: 3pt,
      fill: c-graphite,
      radius: 1pt,
    )
    #v(6pt)
    #text(weight: "bold", size: 12pt, fill: c-black)[
      #counter(heading).display() #it.body
    ]
  ]
  v(0.4em)
}

#show heading.where(level: 3): it => {
  v(0.6em)
  text(weight: "bold", size: 10.5pt, fill: c-graphite)[
    #box(width: 5pt, height: 5pt, fill: c-slate, radius: 100%)
    #h(8pt)
    #counter(heading).display() #it.body
  ]
  v(0.3em)
}

// ── Components ─────────────────────────────────────────────────────────────
#let info-box(title: "", body) = {
  rect(
    width: 100%,
    fill: c-ivory,
    stroke: (left: 3pt + c-slate, rest: 0.8pt + c-platinum),
    radius: (right: 6pt, left: 3pt),
    inset: 12pt,
  )[
    #if title != "" [
      #text(weight: "bold", fill: c-graphite, size: 9.5pt, tracking: 0.5pt)[#upper(title)]
      #v(4pt)
    ]
    #set text(size: 9.5pt)
    #body
  ]
}

#let insight-box(body) = {
  rect(
    width: 100%,
    fill: c-ivory,
    stroke: (left: 3pt + c-black, rest: 0.8pt + c-platinum),
    radius: (right: 6pt, left: 3pt),
    inset: 12pt,
  )[
    #text(weight: "bold", fill: c-black, size: 9.5pt)[Key Insight]
    #v(4pt)
    #set text(size: 9.5pt)
    #body
  ]
}

#let metric-card(label, value, unit: "") = {
  rect(
    width: 100%,
    fill: c-white,
    stroke: 0.8pt + c-platinum,
    radius: 6pt,
    inset: 12pt,
  )[
    #align(center)[
      #text(size: 8pt, fill: c-silver, weight: "bold", tracking: 1pt)[#upper(label)]
      #v(3pt)
      #text(size: 22pt, fill: c-black, weight: "bold")[#value]
      #if unit != "" [
        #text(size: 8pt, fill: c-silver)[#unit]
      ]
    ]
  ]
}

// ── Figure & Table Styling ─────────────────────────────────────────────────
#show figure: it => {
  v(6pt)
  block(
    width: 100%,
    fill: c-white,
    stroke: 0.8pt + c-platinum,
    radius: 6pt,
    inset: 10pt,
  )[
    #align(center)[#it.body]
    #v(6pt)
    #align(center)[
      #text(size: 9pt, fill: c-steel, weight: "medium")[#it.caption]
    ]
  ]
  v(6pt)
}

#show table: it => {
  rect(
    width: 100%,
    fill: c-white,
    stroke: 0.8pt + c-platinum,
    radius: 6pt,
    inset: 0pt,
  )[
    #it
  ]
}

// ═══════════════════════════════════════════════════════════════════════════
//  TITLE PAGE
// ═══════════════════════════════════════════════════════════════════════════
#page(margin: 0pt, header: none, footer: none)[
  #rect(width: 100%, height: 100%, fill: c-black)[
    #v(1fr)
    #align(center)[
      #rect(width: 80pt, height: 2pt, fill: c-silver, radius: 1pt)
      #v(24pt)
      
      #rect(fill: c-charcoal, radius: 4pt, inset: (x: 12pt, y: 6pt), stroke: 0.5pt + c-slate)[
        #text(size: 10pt, fill: c-white, weight: "bold", tracking: 2pt)[TASK-1]
      ]
      #v(16pt)
      
      #text(font: "New Computer Modern", size: 10pt, fill: c-silver, weight: "bold", tracking: 3pt)[
        CN6021 — ADVANCED TOPICS IN AI & DATA SCIENCE
      ]
      #v(16pt)
      
      #text(font: "New Computer Modern", size: 30pt, fill: c-white, weight: "bold")[Customer Churn Prediction]
      #v(8pt)
      #text(font: "New Computer Modern", size: 14pt, fill: c-silver)[
        Using a Shallow Neural Network
      ]
      
      #v(20pt)
      #rect(width: 60pt, height: 1pt, fill: c-graphite)
      #v(16pt)
      
      #text(size: 11pt, fill: c-silver)[Group Coursework Submission]
      #v(6pt)
      #text(size: 10pt, fill: c-steel)[
        Shyam Vijay Jagani (2611208) #h(8pt) | #h(8pt) Jasmi Alasapuri (2571395) \
        Fatema Doctor (2604383) #h(8pt) | #h(8pt) Parth Rathwa (2509367) \
        #v(4pt)
        May 2026
      ]
      
      #v(36pt)
      
      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 14pt,
        inset: (x: 24pt),
        rect(
          fill: c-graphite,
          radius: 8pt,
          inset: 14pt,
          stroke: 0.5pt + c-slate,
        )[
          #align(center)[
            #text(size: 8pt, fill: c-silver, weight: "bold", tracking: 1pt)[AUC-ROC]
            #v(6pt)
            #text(size: 26pt, fill: c-white, weight: "bold")[0.917]
          ]
        ],
        rect(
          fill: c-graphite,
          radius: 8pt,
          inset: 14pt,
          stroke: 0.5pt + c-slate,
        )[
          #align(center)[
            #text(size: 8pt, fill: c-silver, weight: "bold", tracking: 1pt)[MACRO F1]
            #v(6pt)
            #text(size: 26pt, fill: c-white, weight: "bold")[0.860]
          ]
        ],
        rect(
          fill: c-graphite,
          radius: 8pt,
          inset: 14pt,
          stroke: 0.5pt + c-slate,
        )[
          #align(center)[
            #text(size: 8pt, fill: c-silver, weight: "bold", tracking: 1pt)[FEATURES]
            #v(6pt)
            #text(size: 26pt, fill: c-white, weight: "bold")[20]
          ]
        ],
      )
    ]
    #v(1fr)
    
    #rect(width: 100%, height: 3pt, fill: c-graphite)
  ]
]

// Reset heading counter after title page to prevent "0 Contents"
#counter(heading).update(0)

// ── Table of Contents ──────────────────────────────────────────────────────
#page[
  #v(10pt)
  #text(size: 18pt, weight: "bold", fill: c-black)[Table Of Contents]
  #v(4pt)
  #box(width: 50pt, height: 3pt, fill: c-charcoal)
  #v(12pt)
  #outline(title: none, indent: 1.5em, depth: 3)
]

// ═══════════════════════════════════════════════════════════════════════════
= Introduction

Customer churn — the loss of clients or subscribers — represents one of the most costly challenges for subscription-based and e-commerce businesses. Reducing churn by as little as 5% has been shown to increase profitability by 25–95%, making accurate churn prediction a high-value applied machine learning problem. Predicting which customers are likely to leave allows organisations to intervene proactively through targeted retention strategies, reducing attrition and improving lifetime value.

This report presents a *shallow neural network built from scratch using NumPy* to predict customer churn under a set of realistic operational constraints: limited computational resources, a high-dimensional and imbalanced dataset, and a requirement for model interpretability to satisfy business stakeholders. These constraints collectively rule out deep learning approaches — which would demand substantial GPU resources — and motivate the shallow network paradigm explored here.

Several data challenges must be overcome. The dataset exhibits high dimensionality (25 raw features, some redundant), known non-linear relationships between customer behaviour and churn, significant class imbalance (approximately 2.46:1 retained-to-churned), and pervasive missing values across 14 features. Each challenge is addressed explicitly in the methodology.

*Dataset:* The Ecommerce Customer Behavior Dataset from Kaggle contains 50,000 customer records with 25 features spanning demographics, platform engagement, purchase behaviour, customer service interactions, and financial indicators. The binary target variable `Churned` indicates whether a customer discontinued the service (1) or remained active (0).

*Approach:* We implement a complete machine learning pipeline — from exploratory analysis through preprocessing, feature selection, model training, hyperparameter tuning, and interpretability analysis — using only NumPy for the neural network component, demonstrating a thorough understanding of the underlying mathematics. The Universal Approximation Theorem guarantees that a single hidden layer with a sufficient number of units and a non-linear activation function can approximate any continuous function to arbitrary precision, providing the theoretical justification for the shallow architecture adopted here.

= Methodology

== Data Loading and Exploratory Data Analysis

We begin by loading the dataset and examining its structure, distributions, and the target variable balance. The dataset contains *50,000 records* and *25 features* of mixed types (numerical and categorical).

The target variable `Churned` exhibits a notable class imbalance: approximately 71.1% of customers are retained while 28.9% have churned, yielding a 2.46:1 ratio.

#figure(
  image("outputs/01_target_distribution.png", width: 70%),
  caption: [Target Variable Distribution — 71.1% retained vs 28.9% churned.],
)

=== Missing Values

Several features contain significant missing data, with `Social_Media_Engagement_Score` having the highest at 6,000 (12%) missing values. This necessitates careful imputation during preprocessing.

=== Correlation Analysis

The correlation heatmap reveals the relationships between numerical features. No pairs exhibit extremely high collinearity ($|r| > 0.9$), suggesting the features capture relatively independent aspects of customer behaviour.

#figure(
  image("outputs/02_correlation_heatmap.png", width: 90%),
  caption: [Correlation Heatmap of Numerical Features.],
)

=== Feature Distributions by Churn Status

Visualising feature distributions split by churn status helps identify potential predictive signals. Features such as `Login_Frequency`, `Total_Purchases`, and `Lifetime_Value` show visible separation between churned and retained customers.

#figure(
  image("outputs/03_feature_distributions.png", width: 100%),
  caption: [Feature Distributions by Churn Status.],
)

=== PCA Visualisation

We project the data onto two principal components purely for visualisation, to assess the separability of the two classes in a reduced space. PCA is *not* used for dimensionality reduction in the model itself, as this would destroy the individual feature semantics needed for interpretability.

#figure(
  image("outputs/04_pca_scatter.png", width: 70%),
  caption: [PCA — 2D Projection Coloured by Churn Status.],
)

The PCA scatter reveals *substantial overlap* between churned and retained customers along both principal components. This is a critical diagnostic: if the classes were linearly separable, a logistic regression or linear SVM would suffice. The observed overlap confirms that the decision boundary is non-linear, directly motivating the use of a hidden layer with a non-linear activation function.

== Data Preprocessing

The preprocessing pipeline addresses several data quality challenges identified during the EDA:
+ *High Cardinality Categoricals*: We drop the `City` feature (40 unique values) to prevent excessive dimensionality explosion during one-hot encoding.
+ *Outlier Handling*: We cap extreme outliers (e.g. `Age` > 100).
+ *Missing Value Imputation*: We use median imputation for numerical features (robust to outliers) and mode imputation for categorical features.
+ *Encoding*: We apply standard one-hot encoding for nominal categories and ordinal mapping for ordinal features like `Subscription_Level`.
+ *Standardisation*: We fit a `StandardScaler` on the training set and transform both train and test sets to ensure zero mean and unit variance for neural network stability.

== Feature Selection and Dimensionality Reduction

To handle high dimensionality, we implement feature selection using *Mutual Information*.

We compute the mutual information score between each preprocessed feature and the target variable. This metric captures both linear and non-linear dependencies. We drop the bottom 5 features with the lowest MI scores, reducing the final feature space from 25 to 20 dimensions, thereby stripping out noise and mitigating the curse of dimensionality.

#figure(
  image("outputs/05_mutual_information.png", width: 80%),
  caption: [Mutual Information Feature Importance Ranking.],
)

== Addressing Class Imbalance

We address the 2.46:1 class imbalance using a *Weighted Loss Function*. We implement a weighted Binary Cross-Entropy (BCE) loss:

$ cal(L)_"weighted BCE" = - 1/N sum_(i=1)^N [w_"pos" dot y_i log(hat(y)_i) + w_"neg" dot (1 - y_i) log(1 - hat(y)_i)] $

We compute the weights dynamically based on class frequencies. By heavily penalising misclassifications of the minority (churned) class, the network is forced to learn its distinguishing features rather than collapsing to the majority class prior.

== Shallow Neural Network Architecture

We implemented a custom shallow neural network class using NumPy, consisting of:
- Input layer: 20 neurons (features)
- Hidden layer: 32 units with ReLU activation
- Output layer: 1 unit with Sigmoid activation

*Handling Non-Linear Relationships:* The ReLU hidden layer introduces non-linearities, allowing the network to fold and warp the feature space to separate the overlapping classes seen in the PCA plot. Without this hidden layer, the network would reduce to standard logistic regression, which we established is inadequate for this dataset.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: none,
    table.hline(stroke: 2pt + c-charcoal),
    table.header([*Component*], [*Choice*], [*Rationale*]),
    table.hline(stroke: 0.5pt + c-platinum),
    [Hidden activation], [ReLU], [Handles non-linear relationships; avoids vanishing gradients],
    [Output activation], [Sigmoid], [Maps to [0,1] probability for binary classification],
    [Weight init], [He initialisation], [Maintains variance for ReLU; prevents dead neurons],
    [Loss], [Weighted BCE], [Addresses class imbalance directly in the objective],
    [Optimiser], [Mini-batch SGD], [Balances convergence speed and gradient stability],
    [Regularisation], [L2 (weight decay)], [Prevents overfitting on high-dimensional features],
    [Early stopping], [Patience-based], [Stops training when validation loss plateaus],
    table.hline(stroke: 2pt + c-charcoal),
  ),
  caption: [Architectural Design Decisions.],
)

== Hyperparameter Tuning

We perform a grid search over key hyperparameters (hidden units, learning rate, and L2 lambda). The optimal configuration found was *32 hidden units, learning rate = 0.05, L2 $lambda$ = 0.001*. A smaller hidden layer outperforms larger ones, preventing overfitting on the validation set.

== Training the Final Model

The final model is trained on the training set with the optimal hyperparameters, monitored on a fixed validation set for early stopping.

#figure(
  image("outputs/06_training_curves.png", width: 100%),
  caption: [Training and Validation Loss/Accuracy Curves.],
)

= Results

== Evaluation Metrics

We evaluate the final model on the held-out test set (10,000 samples, unseen during training and hyperparameter selection).

#grid(
  columns: (1fr, 1fr),
  column-gutter: 12pt,
  figure(
    image("outputs/07_confusion_matrix.png", width: 100%),
    caption: [Confusion Matrix on Test Set.],
  ),
  figure(
    image("outputs/08_roc_curve.png", width: 100%),
    caption: [ROC Curve.],
  )
)

The model achieves an *AUC-ROC of 0.9172*, indicating excellent discriminative ability. The high recall for the churned class (85%) demonstrates that the weighted loss function successfully addressed the class imbalance.

== Precision-Recall Analysis and Threshold Optimisation

A fixed threshold of 0.5 is not necessarily optimal under class imbalance. We analyse the precision-recall trade-off to identify the operating point that maximises F1-score.

#figure(
  image("outputs/09_pr_curve_threshold.png", width: 100%),
  caption: [Precision-Recall Curve with Optimal F1 Threshold.],
)

Applying the optimal threshold significantly improves the F1 score for the minority class, ensuring the business intervenes at the most cost-effective probability boundary.

== Interpretability Analysis

Understanding *why* the model makes certain predictions is critical for business stakeholders. We employ two complementary methods:

1. *Weight-based importance*: Computes $|W_1| dot |W_2|$ — the aggregated absolute weight magnitude path from each input feature through the hidden layer to the output.
2. *Permutation importance*: Shuffles feature values and measures the resulting drop in AUC-ROC. This captures the true contribution of each feature to predictive performance, including non-linear interactions.

#figure(
  image("outputs/10_feature_importance.png", width: 100%),
  caption: [Feature Importance — Weight-Based (left) and Permutation (right).],
)

*Key insights:*
- *Lifetime_Value* is the most influential feature. Customers with below-average LTV are substantially more likely to churn.
- *Customer_Service_Calls* is the second strongest signal, reflecting unresolved grievances.
- *Cart_Abandonment_Rate* acts as a strong early-warning behavioural indicator.

= Conclusions

#block(
  width: 100%,
  fill: c-ivory,
  radius: 8pt,
  inset: 18pt,
  stroke: (top: 0.5pt + c-charcoal, bottom: 0.5pt + c-slate),
  breakable: true,
)[
#text(size: 12pt, weight: "bold", fill: c-black)[Executive Summary]
#v(8pt)
We developed an interpretable, computationally lightweight churn prediction system meeting all coursework constraints:

#v(6pt)
#grid(
  columns: (1fr, 1fr),
  column-gutter: 16pt,
  row-gutter: 10pt,
  [✓ *AUC-ROC: 0.917* and *Macro F1: 0.860*],
  [✓ *High-dimensionality addressed* via MI selection],
  [✓ *Class imbalance solved* via Weighted BCE loss],
  [✓ *Non-linearity handled* via ReLU hidden layer],
  [✓ *Built entirely in NumPy* for compute constraints],
  [✓ *Highly interpretable* via Permutation Importance],
)
]

The pipeline demonstrates that careful preprocessing, weighted loss functions, and disciplined hyperparameter tuning can yield highly performant and interpretable models without resorting to deep learning.

== Limitations and Future Work

+ *Shallow network ceiling*: A single hidden layer may still underfit genuinely high-order feature interactions. Gradient-boosted trees (e.g., XGBoost) might outperform this model.
+ *Interpretability method limits*: Permutation importance measures marginal importance. SHAP values would provide per-prediction, signed feature attributions for production deployment.
+ *Threshold choice sensitivity*: The optimal threshold was selected to maximise F1, but businesses may prefer to maximise recall or precision based on specific cost ratios of interventions.

