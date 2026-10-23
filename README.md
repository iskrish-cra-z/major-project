# Diabetes Risk Prediction

A machine learning project that predicts diabetes risk using the **Pima Indians Diabetes Database**.

## Business Problem
A rural health clinic wants a quick screening tool that flags patients who are at high risk of diabetes, so a doctor can prioritise them for a full blood test. The clinic can afford to test only **30% of walk-ins per day**.

## Dataset
- **Source**: [Pima Indians Diabetes Database – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (1 = Diabetic, 0 = Not Diabetic)

## What This Project Covers
- Loading a real CSV and inspecting it
- Handling hidden missing values (zeros replaced with median)
- Feature scaling with `StandardScaler`
- Stratified train-test split (80/20)
- Training **KNN**, **Logistic Regression**, and **Decision Tree**
- Evaluating with Accuracy, Precision, Recall, F1, and Confusion Matrix
- Top-30% threshold strategy (business-driven, not just 0.5 cutoff)
- Picking the best model based on **Recall** (maximising catch of real diabetics)

## How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Add the dataset
Place `diabetes.csv` in the project folder (download from Kaggle link above).

### 3. Run the script
```bash
python diabetes_prediction.py
```

## Output
The script prints step-by-step results and saves 4 charts:
| File | Description |
|---|---|
| `correlation_heatmap.png` | Feature correlation matrix |
| `feature_distributions.png` | KDE plots per class |
| `model_comparison.png` | Accuracy / Recall / F1 bar chart |
| `confusion_matrix.png` | Final confusion matrix heatmap |

## Models Compared
| Model | Ranked by |
|---|---|
| K-Nearest Neighbours | Recall + F1 |
| Logistic Regression | Recall + F1 |
| Decision Tree | Recall + F1 |
