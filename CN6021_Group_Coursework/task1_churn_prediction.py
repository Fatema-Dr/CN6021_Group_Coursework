"""
CN6021 — Advanced Topics in AI and Data Science: Coursework Task 1
=================================================================
Customer Churn Prediction using a Shallow Neural Network (Pure NumPy)

Dataset : Ecommerce Customer Behavior Dataset (Kaggle)
Target  : Churned (0 = Active, 1 = Churned)

This script covers:
  1.  Data Loading & Exploratory Data Analysis
  2.  Data Preprocessing (missing values, outliers, encoding, scaling)
  3.  Feature Selection / Dimensionality Reduction
  4.  Class Imbalance Handling (weighted loss)
  5.  Shallow Neural Network from scratch in NumPy
  6.  Hyperparameter Tuning
  7.  Evaluation (Precision, Recall, F1, AUC-ROC)
  8.  Interpretability (weight-based + permutation importance)
  9.  Visualisations & Screenshots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "ecommerce_customer_churn_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════
# STEP 1 — Data Loading & Exploratory Data Analysis
# ═══════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Data Loading & Exploratory Data Analysis")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Churned'].value_counts()}")
print(f"Churn rate: {df['Churned'].mean()*100:.1f}%")

# --- Plot: Target distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Churned"].value_counts()
bars = ax.bar(["Retained (0)", "Churned (1)"], counts.values,
              color=["#2ecc71", "#e74c3c"], edgecolor="black")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            str(val), ha="center", fontweight="bold")
ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=150)
plt.close()

# --- Plot: Correlation heatmap (numerical features only) ---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numerical_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.close()

# --- Plot: Distribution of key numerical features ---
key_features = ["Age", "Membership_Years", "Login_Frequency",
                "Total_Purchases", "Lifetime_Value", "Credit_Balance"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, feat in zip(axes.flat, key_features):
    for label, colour in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = df[df["Churned"] == label][feat].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=f"{'Retained' if label==0 else 'Churned'}",
                color=colour, edgecolor="black", linewidth=0.5)
    ax.set_title(feat, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
plt.suptitle("Feature Distributions by Churn Status", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_distributions.png"), dpi=150)
plt.close()

print("\n✓ EDA plots saved to outputs/")

# ═══════════════════════════════════════════════
# STEP 2 — Data Preprocessing
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing")
print("=" * 70)

# 2a. Drop 'City' — too many unique values (40) for one-hot encoding
df.drop(columns=["City"], inplace=True)
print("Dropped 'City' column (40 unique values — too high cardinality)")

# 2b. Handle outliers
#  - Age max = 200 → cap at 100
#  - Total_Purchases has negatives → clip to 0
df["Age"] = df["Age"].clip(upper=100)
df["Total_Purchases"] = df["Total_Purchases"].clip(lower=0)
print("Clipped Age (max 100), Total_Purchases (min 0)")

# 2c. Impute missing values
#  Numerical: median imputation
#  Categorical: mode imputation
num_cols = df.select_dtypes(include=[np.number]).columns.drop("Churned").tolist()
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

print(f"Imputed missing values — numerical (median), categorical (mode)")
print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# 2d. One-hot encode categorical features
print(f"\nCategorical columns to encode: {cat_cols}")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"Shape after one-hot encoding: {df.shape}")

# 2e. Separate features and target
y = df["Churned"].values
X = df.drop(columns=["Churned"]).values
feature_names = df.drop(columns=["Churned"]).columns.tolist()
print(f"Features: {len(feature_names)} | Samples: {X.shape[0]}")

# 2f. Train / test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 2g. Feature scaling (StandardScaler — fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("Applied StandardScaler (fit on train, transform on test)")

# ═══════════════════════════════════════════════
# STEP 3 — Feature Selection / Dimensionality Reduction
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Feature Selection / Dimensionality Reduction")
print("=" * 70)

# 3a. Correlation-based removal (|r| > 0.90 with another feature)
corr_matrix = np.corrcoef(X_train, rowvar=False)
n_feats = corr_matrix.shape[0]
to_drop_idx = set()
for i in range(n_feats):
    for j in range(i + 1, n_feats):
        if abs(corr_matrix[i, j]) > 0.90:
            # Drop the feature with a smaller mean |correlation| with target
            to_drop_idx.add(j)

if to_drop_idx:
    dropped_names = [feature_names[i] for i in to_drop_idx]
    print(f"Dropping {len(to_drop_idx)} features due to high multicollinearity (|r|>0.9): {dropped_names}")
    keep_idx = [i for i in range(n_feats) if i not in to_drop_idx]
    X_train = X_train[:, keep_idx]
    X_test  = X_test[:, keep_idx]
    feature_names = [feature_names[i] for i in keep_idx]
else:
    print("No features removed due to multicollinearity.")

# 3b. Mutual Information scores
mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({"Feature": feature_names, "MI_Score": mi_scores}).sort_values(
    "MI_Score", ascending=False
)
print(f"\nMutual Information Scores:\n{mi_df.to_string(index=False)}")

# --- Plot: MI scores ---
fig, ax = plt.subplots(figsize=(10, 7))
mi_df_sorted = mi_df.sort_values("MI_Score", ascending=True)
bars = ax.barh(mi_df_sorted["Feature"], mi_df_sorted["MI_Score"],
               color="#3498db", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance — Mutual Information with Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_mutual_information.png"), dpi=150)
plt.close()

# Select features with MI > 0 (keep all informative features)
selected_mask = mi_scores > 0.001
n_selected = selected_mask.sum()
print(f"\nSelected {n_selected}/{len(feature_names)} features with MI > 0.001")
X_train = X_train[:, selected_mask]
X_test  = X_test[:, selected_mask]
feature_names = [f for f, keep in zip(feature_names, selected_mask) if keep]

# 3c. PCA visualisation (2 components — for visualisation only)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_train)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
                     cmap="RdYlGn_r", alpha=0.3, s=8, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA — 2D Projection Coloured by Churn Status", fontsize=13, fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Retained", "Churned"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_pca_scatter.png"), dpi=150)
plt.close()

print(f"\nFinal feature count: {X_train.shape[1]}")
print(f"Final feature names: {feature_names}")

# ═══════════════════════════════════════════════
# STEP 4 — Class Imbalance Handling (Weighted Loss)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Class Imbalance Handling")
print("=" * 70)

n_total   = len(y_train)
n_pos     = y_train.sum()                   # churned (minority)
n_neg     = n_total - n_pos                 # retained (majority)
w_pos     = n_total / (2 * n_pos)           # higher weight for minority
w_neg     = n_total / (2 * n_neg)           # lower weight for majority
print(f"Class weights  — Retained (0): {w_neg:.4f} | Churned (1): {w_pos:.4f}")
print(f"  (Churned class gets {w_pos/w_neg:.2f}x the weight of retained class)")


# ═══════════════════════════════════════════════
# STEP 5 — Shallow Neural Network (Pure NumPy)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Shallow Neural Network — NumPy Implementation")
print("=" * 70)


# ---- Activation Functions ----
def relu(z):
    """ReLU activation — introduces non-linearity, avoids vanishing gradients."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU for backpropagation."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid activation — maps output to [0,1] probability."""
    z = np.clip(z, -500, 500)               # prevent overflow
    return 1.0 / (1.0 + np.exp(-z))


# ---- Loss Function ----
def weighted_bce_loss(y_true, y_pred, w_pos, w_neg):
    """
    Weighted Binary Cross-Entropy Loss.
    Assigns higher weight to the minority class (churned) to handle imbalance.
    """
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    weights = np.where(y_true == 1, w_pos, w_neg)
    loss = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    return loss


# ---- Neural Network Class ----
class ShallowNeuralNetwork:
    """
    A single hidden-layer neural network implemented from scratch in NumPy.

    Architecture:
        Input (n_features) → Hidden (n_hidden, ReLU) → Output (1, Sigmoid)

    Design Justifications:
      - ReLU in hidden layer: handles non-linear relationships, avoids vanishing
        gradient problem, computationally efficient.
      - Sigmoid in output layer: produces probability for binary classification.
      - He initialisation: optimal for ReLU activations.
      - L2 regularisation: prevents overfitting on high-dimensional data.
      - Mini-batch gradient descent: balances convergence speed & stability.
    """

    def __init__(self, n_features, n_hidden=64, learning_rate=0.01,
                 l2_lambda=0.001, batch_size=128, epochs=300):
        self.n_features    = n_features
        self.n_hidden      = n_hidden
        self.lr            = learning_rate
        self.l2_lambda     = l2_lambda
        self.batch_size    = batch_size
        self.epochs        = epochs

        # He initialisation (optimal for ReLU)
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, 1))

        # Training history
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

    def forward(self, X):
        """Forward pass through the network."""
        self.Z1 = X @ self.W1 + self.b1          # linear (input → hidden)
        self.A1 = relu(self.Z1)                   # ReLU activation
        self.Z2 = self.A1 @ self.W2 + self.b2     # linear (hidden → output)
        self.A2 = sigmoid(self.Z2)                 # sigmoid output
        return self.A2

    def backward(self, X, y_true, y_pred, w_pos, w_neg):
        """
        Backward pass — compute gradients via chain rule.

        Gradient derivation:
          dL/dZ2 = (ŷ - y) * sample_weight     [cross-entropy + sigmoid derivative]
          dL/dW2 = A1ᵀ · dZ2
          dL/dZ1 = dZ2 · W2ᵀ ⊙ ReLU'(Z1)
          dL/dW1 = Xᵀ · dZ1
        """
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        # Sample weights for imbalanced classes
        sample_weights = np.where(y_true == 1, w_pos, w_neg)

        # Output layer gradients
        dZ2 = (y_pred - y_true) * sample_weights
        dW2 = (self.A1.T @ dZ2) / m + self.l2_lambda * self.W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dZ1 = (dZ2 @ self.W2.T) * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m + self.l2_lambda * self.W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_proba(self, X):
        """Return probability of churning (class 1)."""
        return self.forward(X).flatten()

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            w_pos=1.0, w_neg=1.0, patience=20, verbose=True):
        """
        Train the network using mini-batch gradient descent.

        Args:
            patience: number of epochs without val loss improvement before
                      early stopping (prevents overfitting).
        """
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(self.epochs):
            # Shuffle training data each epoch
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, w_pos, w_neg)

            # Compute epoch metrics
            train_pred = self.forward(X_train)
            train_loss = weighted_bce_loss(y_train.reshape(-1, 1), train_pred, w_pos, w_neg)
            train_acc  = np.mean((train_pred.flatten() >= 0.5) == y_train)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = weighted_bce_loss(y_val.reshape(-1, 1), val_pred, w_pos, w_neg)
                val_acc  = np.mean((val_pred.flatten() >= 0.5) == y_val)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"  Epoch {epoch+1:>4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                print(msg)

        # Restore best weights
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights
            if verbose:
                print(f"  Restored best weights (val loss: {best_val_loss:.4f})")

        return self


# ═══════════════════════════════════════════════
# STEP 6 — Hyperparameter Tuning
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Hyperparameter Tuning")
print("=" * 70)

# Split train set further into train/val for tuning (80/20)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)

# ---- Grid Search ----
param_grid = {
    "n_hidden":      [32, 64, 128],
    "learning_rate": [0.001, 0.01, 0.05],
    "l2_lambda":     [0, 0.001, 0.01],
}

best_f1 = 0
best_params = {}
results = []

print("\nSearching over hyperparameter grid...")
total_combos = (len(param_grid["n_hidden"]) *
                len(param_grid["learning_rate"]) *
                len(param_grid["l2_lambda"]))
combo_idx = 0

for n_h in param_grid["n_hidden"]:
    for lr in param_grid["learning_rate"]:
        for l2 in param_grid["l2_lambda"]:
            combo_idx += 1
            np.random.seed(RANDOM_STATE)

            model = ShallowNeuralNetwork(
                n_features=X_tr.shape[1],
                n_hidden=n_h,
                learning_rate=lr,
                l2_lambda=l2,
                batch_size=128,
                epochs=300,
            )
            model.fit(X_tr, y_tr, X_val, y_val, w_pos, w_neg,
                      patience=20, verbose=False)

            val_preds = model.predict(X_val)
            f1 = f1_score(y_val, val_preds)
            auc = roc_auc_score(y_val, model.predict_proba(X_val))

            results.append({
                "n_hidden": n_h, "lr": lr, "l2_lambda": l2,
                "val_f1": f1, "val_auc": auc
            })

            if f1 > best_f1:
                best_f1 = f1
                best_params = {"n_hidden": n_h, "learning_rate": lr, "l2_lambda": l2}

            if combo_idx % 9 == 0 or combo_idx == total_combos:
                print(f"  [{combo_idx}/{total_combos}] n_h={n_h}, lr={lr}, l2={l2} → "
                      f"F1={f1:.4f}, AUC={auc:.4f}")

results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
print(f"\nTop 5 configurations:\n{results_df.head().to_string(index=False)}")
print(f"\n★ Best hyperparameters: {best_params}")

# ═══════════════════════════════════════════════
# Train final model with best hyperparameters
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("Training final model with best hyperparameters...")
print("=" * 70)

np.random.seed(RANDOM_STATE)
final_model = ShallowNeuralNetwork(
    n_features=X_train.shape[1],
    n_hidden=best_params["n_hidden"],
    learning_rate=best_params["learning_rate"],
    l2_lambda=best_params["l2_lambda"],
    batch_size=128,
    epochs=500,
)

# Use a small held-out portion for early stopping
X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
)
final_model.fit(X_tr_final, y_tr_final, X_val_final, y_val_final,
                w_pos, w_neg, patience=30, verbose=True)

# --- Plot: Training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(final_model.train_losses, label="Train Loss", color="#2c3e50", linewidth=1.5)
ax1.plot(final_model.val_losses, label="Val Loss", color="#e74c3c", linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Weighted BCE Loss")
ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(final_model.train_accs, label="Train Accuracy", color="#2c3e50", linewidth=1.5)
ax2.plot(final_model.val_accs, label="Val Accuracy", color="#e74c3c", linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_training_curves.png"), dpi=150)
plt.close()


# ═══════════════════════════════════════════════
# STEP 7 — Evaluation on Test Set
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Evaluation on Test Set")
print("=" * 70)

y_pred_proba = final_model.predict_proba(X_test)
y_pred       = final_model.predict(X_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_score:.4f}")

# Precision, Recall, F1 (macro)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
print(f"Macro — Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# --- Plot: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
            annot_kws={"size": 14})
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix (AUC-ROC: {auc_score:.4f})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrix.png"), dpi=150)
plt.close()

# --- Plot: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2c3e50", linewidth=2, label=f"Shallow NN (AUC = {auc_score:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random Baseline")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_roc_curve.png"), dpi=150)
plt.close()


# ═══════════════════════════════════════════════
# STEP 8 — Interpretability Analysis
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Interpretability Analysis")
print("=" * 70)

# 8a. Weight-based feature importance
#     Aggregate absolute weights: |W1| · |W2| gives each input feature's
#     contribution through the hidden layer to the output.
weight_importance = np.abs(final_model.W1) @ np.abs(final_model.W2)
weight_importance = weight_importance.flatten()
weight_importance = weight_importance / weight_importance.sum()   # normalise

# 8b. Permutation importance
#     For each feature, shuffle its values and measure how much AUC-ROC drops.
print("\nComputing permutation importance (this may take a moment)...")
baseline_auc = roc_auc_score(y_test, final_model.predict_proba(X_test))
perm_importance = np.zeros(X_test.shape[1])

for i in range(X_test.shape[1]):
    auc_drops = []
    for _ in range(5):                       # 5 repetitions for stability
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, i])
        perm_auc = roc_auc_score(y_test, final_model.predict_proba(X_perm))
        auc_drops.append(baseline_auc - perm_auc)
    perm_importance[i] = np.mean(auc_drops)

# --- Plot: Feature Importance (dual view) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Weight-based
idx_w = np.argsort(weight_importance)
ax1.barh([feature_names[i] for i in idx_w], weight_importance[idx_w],
         color="#3498db", edgecolor="black", linewidth=0.5)
ax1.set_xlabel("Normalised Weight Importance")
ax1.set_title("Weight-Based Feature Importance\n(|W₁| · |W₂|)", fontsize=13, fontweight="bold")

# Permutation-based
idx_p = np.argsort(perm_importance)
colours = ["#e74c3c" if v > 0 else "#95a5a6" for v in perm_importance[idx_p]]
ax2.barh([feature_names[i] for i in idx_p], perm_importance[idx_p],
         color=colours, edgecolor="black", linewidth=0.5)
ax2.set_xlabel("Mean AUC-ROC Drop (higher = more important)")
ax2.set_title("Permutation Feature Importance\n(AUC-ROC decrease when shuffled)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_feature_importance.png"), dpi=150)
plt.close()

# Print top 10 features
print("\nTop 10 Features by Weight Importance:")
for rank, i in enumerate(np.argsort(weight_importance)[::-1][:10], 1):
    print(f"  {rank}. {feature_names[i]:35s} — {weight_importance[i]:.4f}")

print("\nTop 10 Features by Permutation Importance:")
for rank, i in enumerate(np.argsort(perm_importance)[::-1][:10], 1):
    print(f"  {rank}. {feature_names[i]:35s} — AUC drop: {perm_importance[i]:.4f}")


# ═══════════════════════════════════════════════
# STEP 9 — Summary & Final Output
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Summary")
print("=" * 70)

print(f"""
Model Architecture:
  Input → {best_params['n_hidden']} hidden units (ReLU) → 1 output (Sigmoid)

Best Hyperparameters:
  Hidden units   : {best_params['n_hidden']}
  Learning rate  : {best_params['learning_rate']}
  L2 lambda      : {best_params['l2_lambda']}
  Batch size     : 128
  Early stopping : patience = 30

Test Set Performance:
  AUC-ROC   : {auc_score:.4f}
  Precision : {precision:.4f} (macro)
  Recall    : {recall:.4f} (macro)
  F1-Score  : {f1:.4f} (macro)

Outputs saved to: {OUTPUT_DIR}/
  01_target_distribution.png
  02_correlation_heatmap.png
  03_feature_distributions.png
  04_mutual_information.png
  05_pca_scatter.png
  06_training_curves.png
  07_confusion_matrix.png
  08_roc_curve.png
  09_feature_importance.png
""")

print("✓ Task 1 complete!")
