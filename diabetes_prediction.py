
# =============================================================================
#  DIABETES RISK PREDICTION
#  Dataset : Pima Indians Diabetes Database (Kaggle)
#  Goal    : Flag top-30% highest-risk walk-ins for a full blood test
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
)

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Loading Dataset")
print("=" * 60)

# Place 'diabetes.csv' in the same folder as this script.
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
df = pd.read_csv("diabetes.csv")

print(f"\nShape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# ── 3. INSPECT & HANDLE HIDDEN MISSING VALUES (ZEROS) ────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Handling Hidden Missing Values (Zeros)")
print("=" * 60)

# Medically, zero is impossible for these columns -> treat as NaN
zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero counts before replacement:")
for col in zero_not_allowed:
    n_zeros = (df[col] == 0).sum()
    print(f"  {col:30s}: {n_zeros} zeros")

# Replace zeros with NaN
df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

# Fill NaN with the column median (robust to outliers)
df[zero_not_allowed] = df[zero_not_allowed].fillna(df[zero_not_allowed].median())

print("\nMissing values after imputation:", df.isnull().sum().sum())

# ── 4. EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — Exploratory Data Analysis")
print("=" * 60)

print("\nClass distribution:")
print(df["Outcome"].value_counts())
print(f"  Diabetic rate: {df['Outcome'].mean():.1%}")

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# Distribution of each feature by outcome
features = [c for c in df.columns if c != "Outcome"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    df.groupby("Outcome")[col].plot(
        kind="kde", ax=axes[i], legend=(i == 0), title=col
    )
    axes[i].set_xlabel("")
plt.suptitle("Feature Distributions by Outcome (0=Not Diabetic, 1=Diabetic)", y=1.01)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150)
plt.show()

# ── 5. FEATURE / TARGET SPLIT ─────────────────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ── 6. TRAIN-TEST SPLIT (STRATIFIED) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — Train / Test Split (stratified, 80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Train size : {len(X_train)} samples")
print(f"  Test  size : {len(X_test)} samples")
print(f"  Diabetic % in train : {y_train.mean():.1%}")
print(f"  Diabetic % in test  : {y_test.mean():.1%}")

# ── 7. FEATURE SCALING ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — Feature Scaling (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)         # transform test with same scaler

print("\n  Scaling complete. Mean ~0, Std ~1 for all features.")

# ── 8. TRAIN MODELS ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — Training Models")
print("=" * 60)

models = {
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Standard 0.5-threshold predictions (for comparison)
    y_pred = model.predict(X_test_scaled)

    recall   = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    accuracy = (y_pred == y_test).mean()

    results[name] = {
        "model"   : model,
        "recall"  : recall,
        "f1"      : f1,
        "accuracy": accuracy,
    }

    print(f"\n  {name}")
    print(f"    Accuracy : {accuracy:.3f}")
    print(f"    Recall   : {recall:.3f}   <- catching real diabetics")
    print(f"    F1-Score : {f1:.3f}")

# ── 9. MODEL COMPARISON ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — Model Comparison (Business Criteria)")
print("=" * 60)

print("""
  Business goal : maximise RECALL (catch real diabetics) while
                  keeping false alarms manageable.
  We rank by Recall first, F1 second.
""")

comparison_df = pd.DataFrame(
    {k: {"Accuracy": v["accuracy"], "Recall": v["recall"], "F1": v["f1"]}
     for k, v in results.items()}
).T.sort_values(["Recall", "F1"], ascending=False)

print(comparison_df.to_string())

# Bar chart comparison
comparison_df[["Accuracy", "Recall", "F1"]].plot(
    kind="bar", figsize=(9, 5), colormap="viridis", edgecolor="white"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# ── 10. PICK BEST MODEL ───────────────────────────────────────────────────────
best_model_name = comparison_df.index[0].strip()
best_model      = results[comparison_df.index[0]]["model"]

print(f"\n  Best model selected: {best_model_name}")

# ── 11. FINAL EVALUATION — TOP-30% THRESHOLD ─────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 — Final Evaluation (Top-30% Threshold Strategy)")
print("=" * 60)

print("Selected model:", best_model_name)

# Predicted probabilities for the positive class (Diabetic)
probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

# Clinic can test only 30% of walk-ins per day
limit = int(np.ceil(len(X_test) * 0.30))

# Select indices of the top-30% highest-risk patients
selected_indices = np.argsort(probabilities)[-limit:]

# Build final prediction array
final_predictions = np.zeros(len(X_test), dtype=int)
final_predictions[selected_indices] = 1

print("Patients in test set       :", len(X_test))
print("Patients selected for testing:", limit)

print("\nFinal evaluation:")
print(classification_report(
    y_test, final_predictions, zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, final_predictions)
print(cm)

# ── 12. CONFUSION MATRIX HEATMAP ─────────────────────────────────────────────
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Diabetic", "Diabetic"],
    yticklabels=["Not Diabetic", "Diabetic"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Diabetes Prediction — Confusion Matrix\n({best_model_name}, Top-30% Threshold)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\n" + "=" * 60)
print("  DONE — All charts saved as PNG files.")
print("=" * 60)
