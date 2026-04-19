"""
task1_churn_prediction.py
=========================
CN6021 — Advanced Topics in AI and Data Science
Task 1: Customer Churn Prediction Using a Shallow Neural Network

Pipeline
--------
1.  Data loading & EDA
2.  Preprocessing (outlier handling, imputation, encoding, 3-way split, scaling)
3.  Feature selection (correlation filter + mutual information)
4.  Class-imbalance handling (weighted BCE loss)
5.  Shallow neural network (single hidden layer, pure NumPy)
6.  Hyperparameter grid search (27 configs, fixed val set)
7.  Final model training with early stopping
8.  Evaluation (classification report, AUC-ROC, confusion matrix, ROC curve)
9.  Threshold optimisation + precision-recall analysis
10. Interpretability (weight-based + permutation importance, risk profiles)

All figures are saved to outputs/  — set SHOW_PLOTS = True to display them
interactively as well.

Usage
-----
    python task1_churn_prediction.py

Expected data path: data/ecommerce_customer_churn_dataset.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SHOW_PLOTS = False  # Set True to display figures interactively
OUTPUT_DIR = "outputs"
DATA_PATH = "data/ecommerce_customer_churn_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    f1_score,
)

np.random.seed(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save (and optionally show) a figure
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Activation functions & loss
# ─────────────────────────────────────────────────────────────────────────────
def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation — introduces non-linearity, avoids vanishing gradients."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU for backpropagation."""
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid — maps output to [0, 1] churn probability."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def weighted_bce_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w_pos: float,
    w_neg: float,
) -> float:
    """
    Weighted Binary Cross-Entropy.
    Assigns higher penalty to misclassifying the minority (churned) class.
    """
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    w = np.where(y_true == 1, w_pos, w_neg)
    return -np.mean(w * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Shallow Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class ShallowNeuralNetwork:
    """
    Single hidden-layer neural network implemented in pure NumPy.

    Architecture
    ------------
    Input (n features)  →  Hidden layer (n_hidden units, ReLU)
                        →  Output (1 unit, Sigmoid)

    Design choices
    --------------
    - ReLU hidden activation : captures non-linear churn patterns; avoids
                                vanishing gradients (cf. tanh / sigmoid)
    - He initialisation      : maintains variance for ReLU; prevents dead neurons
    - Weighted BCE loss       : addresses class imbalance at the objective level
    - Mini-batch SGD          : balances convergence speed and gradient stability
    - L2 regularisation      : prevents overfitting on high-dimensional inputs
    - Early stopping          : halts training when validation loss plateaus
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 64,
        learning_rate: float = 0.01,
        l2_lambda: float = 0.001,
        batch_size: int = 128,
        epochs: int = 300,
    ) -> None:
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.epochs = epochs

        # He initialisation: var = 2 / n_in
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, 1))

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """X → Z1 → A1(ReLU) → Z2 → A2(Sigmoid)"""
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    # ── Backward pass (chain rule) ────────────────────────────────────────────
    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        w_pos: float,
        w_neg: float,
    ) -> None:
        """
        Gradients via chain rule:
          dL/dZ2 = (ŷ - y) * sample_weight
          dL/dW2 = A1ᵀ · dZ2  +  λ·W2
          dL/dZ1 = dZ2 · W2ᵀ  ⊙  ReLU'(Z1)
          dL/dW1 = Xᵀ  · dZ1  +  λ·W1
        """
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)
        sw = np.where(y_true == 1, w_pos, w_neg)

        dZ2 = (y_pred - y_true) * sw
        dW2 = (self.A1.T @ dZ2) / m + self.l2_lambda * self.W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dZ1 = (dZ2 @ self.W2.T) * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m + self.l2_lambda * self.W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Training loop ──────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        w_pos: float = 1.0,
        w_neg: float = 1.0,
        patience: int = 20,
        verbose: bool = True,
    ) -> "ShallowNeuralNetwork":
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(self.epochs):
            # Mini-batch SGD with shuffle
            idx = np.random.permutation(X_train.shape[0])
            X_s, y_s = X_train[idx], y_train[idx]

            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                y_pred = self.forward(X_s[start:end])
                self.backward(X_s[start:end], y_s[start:end], y_pred, w_pos, w_neg)

            # Track train metrics
            tp = self.forward(X_train)
            tl = weighted_bce_loss(y_train.reshape(-1, 1), tp, w_pos, w_neg)
            ta = np.mean((tp.flatten() >= 0.5) == y_train)
            self.train_losses.append(tl)
            self.train_accs.append(ta)

            # Track val metrics + early stopping
            if X_val is not None:
                vp = self.forward(X_val)
                vl = weighted_bce_loss(y_val.reshape(-1, 1), vp, w_pos, w_neg)
                va = np.mean((vp.flatten() >= 0.5) == y_val)
                self.val_losses.append(vl)
                self.val_accs.append(va)

                if vl < best_val_loss:
                    best_val_loss = vl
                    patience_counter = 0
                    best_weights = (
                        self.W1.copy(),
                        self.b1.copy(),
                        self.W2.copy(),
                        self.b2.copy(),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(
                            f"  Early stopping at epoch {epoch + 1}"
                            f"  (best val loss: {best_val_loss:.4f})"
                        )
                    break

        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights

        return self


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── 4.1  Load data ────────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Loading data & EDA")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape : {df.shape}")
    print(f"Churn rate    : {df['Churned'].mean() * 100:.1f}%")

    # Missing values report
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
    print("\nMissing values (features with >0 missing):")
    print(
        missing_df[missing_df["Missing Count"] > 0]
        .sort_values("Missing Count", ascending=False)
        .to_string()
    )

    # Fig 1 — Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churned"].value_counts()
    bars = ax.bar(
        ["Retained (0)", "Churned (1)"],
        counts.values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            str(val),
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig("01_target_distribution.png")

    # Fig 2 — Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    save_fig("02_correlation_heatmap.png")

    # Fig 3 — Feature distributions by churn status
    key_features = [
        "Age",
        "Membership_Years",
        "Login_Frequency",
        "Total_Purchases",
        "Lifetime_Value",
        "Credit_Balance",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat in zip(axes.flat, key_features):
        for label, colour in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df["Churned"] == label][feat].dropna()
            ax.hist(
                subset,
                bins=30,
                alpha=0.6,
                label="Retained" if label == 0 else "Churned",
                color=colour,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    plt.suptitle(
        "Feature Distributions by Churn Status", fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_fig("03_feature_distributions.png")

    # Fig 4 — PCA visualisation (EDA only — not used for dimensionality reduction)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_vis = pca.fit_transform(
        StandardScaler().fit_transform(
            df.drop(columns=["Churned"])
            .select_dtypes(include=[np.number])
            .fillna(0)
            .values
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca_vis[:, 0],
        X_pca_vis[:, 1],
        c=df["Churned"].values,
        cmap="RdYlGn_r",
        alpha=0.3,
        s=8,
        edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(
        "PCA — 2D Projection Coloured by Churn Status", fontsize=13, fontweight="bold"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Retained", "Churned"])
    plt.tight_layout()
    save_fig("04_pca_scatter.png")

    # ── 4.2  Preprocessing ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Preprocessing")
    print("=" * 65)

    # Drop high-cardinality column
    df.drop(columns=["City"], inplace=True)

    # Clip outliers
    df["Age"] = df["Age"].clip(upper=100)
    df["Total_Purchases"] = df["Total_Purchases"].clip(lower=0)

    # Impute missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.drop("Churned").tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categorical features (drop_first avoids dummy trap)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separate features and target
    y = df["Churned"].values
    X = df.drop(columns=["Churned"]).values
    feature_names = df.drop(columns=["Churned"]).columns.tolist()

    # Fixed three-way split: 70% train / 15% val / 15% test
    # All splits established once here — never re-split downstream.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Scale (fit on train only — prevents leakage into val/test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Features after preprocessing : {X_train.shape[1]}")
    print(
        f"Train / Val / Test           : "
        f"{X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}"
    )
    print(f"Missing values remaining     : {df.isnull().sum().sum()}")

    # ── 4.3  Feature selection ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Feature selection")
    print("=" * 65)

    # 3a. Correlation-based removal (|r| > 0.90)
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    n_feats = corr_matrix.shape[0]
    to_drop_idx = set()
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            if abs(corr_matrix[i, j]) > 0.90:
                to_drop_idx.add(j)

    if to_drop_idx:
        dropped = [feature_names[i] for i in to_drop_idx]
        keep_idx = [i for i in range(n_feats) if i not in to_drop_idx]
        X_train = X_train[:, keep_idx]
        X_val = X_val[:, keep_idx]
        X_test = X_test[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        print(f"Correlation filter: dropped {len(to_drop_idx)} features: {dropped}")
    else:
        print("Correlation filter: no features removed (no |r| > 0.90 pairs)")

    # 3b. Mutual Information feature selection (MI > 0.001)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"Feature": feature_names, "MI_Score": mi_scores})

    # Fig 5 — MI bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    mi_sorted = mi_df.sort_values("MI_Score", ascending=True)
    ax.barh(
        mi_sorted["Feature"],
        mi_sorted["MI_Score"],
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(
        "Feature Importance — Mutual Information with Target",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig("05_mutual_information.png")

    selected_mask = mi_scores > 0.001
    X_train = X_train[:, selected_mask]
    X_val = X_val[:, selected_mask]
    X_test = X_test[:, selected_mask]
    feature_names = [f for f, keep in zip(feature_names, selected_mask) if keep]
    print(
        f"MI filter: retained {sum(selected_mask)}/{len(selected_mask)} features"
        f" (MI > 0.001)"
    )
    print(f"Final feature count: {X_train.shape[1]}")

    # ── 4.4  Class weights ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Class imbalance (weighted BCE)")
    print("=" * 65)

    n_total = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_total - n_pos
    w_pos = n_total / (2 * n_pos)
    w_neg = n_total / (2 * n_neg)
    print(f"Class weights — Retained: {w_neg:.4f} | Churned: {w_pos:.4f}")
    print(f"Churned class receives {w_pos / w_neg:.2f}× weight of retained class")

    # ── 4.5  Hyperparameter grid search ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5 — Hyperparameter grid search (27 configurations)")
    print("=" * 65)

    param_grid = {
        "n_hidden": [32, 64, 128],
        "learning_rate": [0.001, 0.01, 0.05],
        "l2_lambda": [0, 0.001, 0.01],
    }

    best_f1, best_params, results = 0.0, {}, []
    total = (
        len(param_grid["n_hidden"])
        * len(param_grid["learning_rate"])
        * len(param_grid["l2_lambda"])
    )
    combo = 0

    for n_h in param_grid["n_hidden"]:
        for lr in param_grid["learning_rate"]:
            for l2 in param_grid["l2_lambda"]:
                combo += 1
                print(
                    f"  [{combo:2d}/{total}] n_hidden={n_h:3d}  lr={lr:.3f}"
                    f"  l2={l2:.3f}",
                    end="  ",
                    flush=True,
                )
                np.random.seed(RANDOM_STATE)
                model = ShallowNeuralNetwork(
                    n_features=X_train.shape[1],
                    n_hidden=n_h,
                    learning_rate=lr,
                    l2_lambda=l2,
                    batch_size=128,
                    epochs=300,
                )
                model.fit(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    w_pos,
                    w_neg,
                    patience=20,
                    verbose=False,
                )

                f1_val = f1_score(y_val, model.predict(X_val))
                auc_val = roc_auc_score(y_val, model.predict_proba(X_val))
                print(f"F1={f1_val:.4f}  AUC={auc_val:.4f}")

                results.append(
                    {
                        "Hidden Units": n_h,
                        "LR": lr,
                        "L2": l2,
                        "Val F1": round(f1_val, 4),
                        "Val AUC": round(auc_val, 4),
                    }
                )
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_params = {
                        "n_hidden": n_h,
                        "learning_rate": lr,
                        "l2_lambda": l2,
                    }

    results_df = pd.DataFrame(results).sort_values("Val F1", ascending=False)
    print("\nTop 5 configurations:")
    print(results_df.head().to_string(index=False))
    print(f"\n★  Best params : {best_params}  |  Best Val F1: {best_f1:.4f}")

    # ── 4.6  Train final model ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6 — Training final model")
    print("=" * 65)

    np.random.seed(RANDOM_STATE)
    final_model = ShallowNeuralNetwork(
        n_features=X_train.shape[1],
        n_hidden=best_params["n_hidden"],
        learning_rate=best_params["learning_rate"],
        l2_lambda=best_params["l2_lambda"],
        batch_size=128,
        epochs=500,
    )
    # Use the same fixed val split — no new splits here
    final_model.fit(X_train, y_train, X_val, y_val, w_pos, w_neg, patience=30)

    # Fig 6 — Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(final_model.train_losses, label="Train Loss", color="#2c3e50", lw=1.5)
    ax1.plot(final_model.val_losses, label="Val Loss", color="#e74c3c", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted BCE Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(final_model.train_accs, label="Train Accuracy", color="#2c3e50", lw=1.5)
    ax2.plot(final_model.val_accs, label="Val Accuracy", color="#e74c3c", lw=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("06_training_curves.png")

    # ── 4.7  Evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 7 — Evaluation on held-out test set")
    print("=" * 65)

    y_pred_proba = final_model.predict_proba(X_test)
    y_pred = final_model.predict(X_test)

    print("\nClassification Report (threshold = 0.5):")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    auc = roc_auc_score(y_test, y_pred_proba)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Macro    — Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # Fig 7 — Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix  (AUC-ROC: {auc:.4f})", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_fig("07_confusion_matrix.png")

    # Fig 8 — ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2c3e50", lw=2, label=f"Shallow NN (AUC = {auc:.4f})")
    ax.plot(
        [0, 1], [0, 1], "--", color="gray", lw=1, label="Random Baseline (AUC = 0.5)"
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("08_roc_curve.png")

    # ── 4.8  Threshold optimisation & PR curve ────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 8 — Threshold optimisation & Precision-Recall analysis")
    print("=" * 65)

    # Optimal threshold from val set (not test — avoids leakage)
    val_proba = final_model.predict_proba(X_val)
    prec_val, rec_val, thr = precision_recall_curve(y_val, val_proba)
    f1_thr = 2 * prec_val[:-1] * rec_val[:-1] / (prec_val[:-1] + rec_val[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_thr)
    optimal_threshold = thr[optimal_idx]

    y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)

    # PR curve on test set
    prec_test, rec_test, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    # Fig 9 — PR curve + F1-vs-threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        rec_test, prec_test, color="#2c3e50", lw=2, label=f"Shallow NN (AP = {ap:.4f})"
    )
    axes[0].axhline(
        y_test.mean(),
        color="gray",
        linestyle="--",
        lw=1,
        label=f"Random Baseline (AP = {y_test.mean():.2f})",
    )
    axes[0].set_xlabel("Recall", fontsize=12)
    axes[0].set_ylabel("Precision", fontsize=12)
    axes[0].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thr, f1_thr, color="#3498db", lw=2)
    axes[1].axvline(
        optimal_threshold,
        color="#e74c3c",
        linestyle="--",
        lw=1.5,
        label=f"Optimal threshold = {optimal_threshold:.3f}",
    )
    axes[1].set_xlabel("Decision Threshold", fontsize=12)
    axes[1].set_ylabel("F1 Score (Churned)", fontsize=12)
    axes[1].set_title(
        "F1 Score vs Decision Threshold\n(evaluated on val set)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("09_pr_curve_threshold.png")

    print(
        f"Default  threshold (0.50)  — Test F1 (churn): {f1_score(y_test, y_pred):.4f}"
    )
    print(
        f"Optimal  threshold ({optimal_threshold:.3f}) — Test F1 (churn): "
        f"{f1_score(y_test, y_pred_opt):.4f}"
    )
    print(f"\nClassification Report (optimal threshold = {optimal_threshold:.3f}):")
    print(
        classification_report(y_test, y_pred_opt, target_names=["Retained", "Churned"])
    )

    # ── 4.9  Interpretability ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 9 — Interpretability analysis")
    print("=" * 65)

    # Weight-based importance: |W1| · |W2|  (normalised)
    weight_imp = (np.abs(final_model.W1) @ np.abs(final_model.W2)).flatten()
    weight_imp /= weight_imp.sum()

    # Permutation importance: mean AUC-ROC drop over 5 shuffles per feature
    baseline_auc = roc_auc_score(y_test, final_model.predict_proba(X_test))
    perm_imp = np.zeros(X_test.shape[1])
    for i in range(X_test.shape[1]):
        drops = []
        for _ in range(5):
            X_p = X_test.copy()
            np.random.shuffle(X_p[:, i])
            drops.append(
                baseline_auc - roc_auc_score(y_test, final_model.predict_proba(X_p))
            )
        perm_imp[i] = np.mean(drops)

    # Fig 10 — Feature importance (dual plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    idx_w = np.argsort(weight_imp)
    ax1.barh(
        [feature_names[i] for i in idx_w],
        weight_imp[idx_w],
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("Normalised Weight Importance")
    ax1.set_title(
        "Weight-Based Feature Importance\n(|W₁| · |W₂|)", fontsize=13, fontweight="bold"
    )

    idx_p = np.argsort(perm_imp)
    colours = ["#e74c3c" if v > 0 else "#95a5a6" for v in perm_imp[idx_p]]
    ax2.barh(
        [feature_names[i] for i in idx_p],
        perm_imp[idx_p],
        color=colours,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Mean AUC-ROC Drop (higher = more important)")
    ax2.set_title(
        "Permutation Feature Importance\n(AUC-ROC decrease when shuffled)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    save_fig("10_feature_importance.png")

    # Print top-10 rankings
    print("\nTop 10 Features — Weight Importance:")
    for rank, i in enumerate(np.argsort(weight_imp)[::-1][:10], 1):
        print(f"  {rank:2d}. {feature_names[i]:35s}  {weight_imp[i]:.4f}")

    print("\nTop 10 Features — Permutation Importance:")
    for rank, i in enumerate(np.argsort(perm_imp)[::-1][:10], 1):
        print(f"  {rank:2d}. {feature_names[i]:35s}  AUC drop: {perm_imp[i]:.4f}")

    # Illustrative churn-risk profiles
    feat_means = X_train.mean(axis=0)

    high_risk = feat_means.copy()
    for fname, delta in [
        ("Customer_Service_Calls", +2.0),
        ("Cart_Abandonment_Rate", +2.0),
        ("Lifetime_Value", -2.0),
        ("Days_Since_Last_Purchase", +2.0),
    ]:
        if fname in feature_names:
            high_risk[feature_names.index(fname)] += delta

    low_risk = feat_means.copy()
    for fname, delta in [
        ("Customer_Service_Calls", -1.5),
        ("Cart_Abandonment_Rate", -1.5),
        ("Lifetime_Value", +1.5),
        ("Days_Since_Last_Purchase", -1.5),
    ]:
        if fname in feature_names:
            low_risk[feature_names.index(fname)] += delta

    p_high = final_model.predict_proba(high_risk.reshape(1, -1))[0]
    p_low = final_model.predict_proba(low_risk.reshape(1, -1))[0]

    print("\nIllustrative churn-risk profiles:")
    print(f"  High-risk  (high calls, high abandonment, low LTV) : {p_high:.1%}")
    print(f"  Low-risk   (low calls,  low abandonment,  high LTV): {p_low:.1%}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Pipeline complete.  All figures saved to  '{OUTPUT_DIR}/'")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
