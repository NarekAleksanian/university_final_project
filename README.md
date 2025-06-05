
# Startup Success Classification

## 🧠 Project Overview

This project aims to classify startups as either **Successful** or **Failed** based on a wide range of features—such as funding scores, internet activity, team skills, and market conditions. The classification task is solved using machine learning techniques. After data exploration, cleaning, and feature engineering, we evaluate two models:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

We compare both models and justify our final choice of **Logistic Regression**.

---

## 📊 Dataset

The dataset contains information about multiple startups, with more than 110 features and a binary target variable:  
- `Dependent-Company Status` → with values: `'Success'` or `'Failed'`.

Examples of features:
- `Funding score`
- `Internet activity score`
- `Renown score`
- `Team skill indicators` (e.g., marketing, operations, leadership)
- `Number of direct competitors`
- `Industry trend in investing`
- Categorical features like `Industry`, `Market`, `Location`, etc.

---

## ⚙️ Preprocessing Steps

1. **Missing Values**:
   - Replaced `"No Info"` with `NaN`
   - Numeric columns → filled with **median**
   - Categorical columns → filled with **mode**

2. **Feature Encoding**:
   - Used **one-hot encoding** for categorical features (`drop_first=True`)  
   - Dropped free-text columns like `Company_Name` and `Short Description`

3. **Feature Scaling**:
   - All numeric features were scaled using **StandardScaler** (zero mean, unit variance)

4. **Feature Engineering**:
   - Created a new feature: `Company Age (2025)` = 2025 - `year of founding`

5. **Target Mapping**:
   - `Success` → 1  
   - `Failed` → 0

6. **Train/Test Split**:
   - Used **80/20** split with random state for reproducibility

---

## 📈 Model Comparison

We evaluated **Logistic Regression** and **KNN** using the same preprocessed dataset. Below is a detailed comparison:

| Criteria | Logistic Regression | KNN (k=5) |
|----------|---------------------|-----------|
| **Accuracy** | ✅ Higher (~86%) | ❌ Lower (~78–82%) |
| **Precision** | ✅ High (89–90%) | Moderate (81–85%) |
| **Recall** | ✅ High (88–89%) | Lower (70–80%) |
| **Training Speed** | ✅ Very Fast | ❌ Slower (due to storing all instances) |
| **Interpretability** | ✅ Very clear coefficients | ❌ Opaque ("black box") |
| **Scalability** | ✅ Scales well with features | ❌ Poor for high-dimensional data |
| **Handling of Noise/Outliers** | ✅ More robust | ❌ Sensitive to noise and irrelevant features |
| **Memory Efficiency** | ✅ Model is compact | ❌ Needs full dataset in memory at inference |

---

## ✅ Why Logistic Regression is Better for This Problem

### 1. **Interpretability**

Logistic Regression provides **feature coefficients**, allowing us to understand:
- Which features push predictions toward **Success** or **Failure**
- How **strongly** each feature influences the outcome

This is crucial for **business insight** in startup analysis.

### 2. **Performance Metrics**

Logistic Regression consistently achieves:
- **Higher Accuracy** (~86%)
- **Higher Precision** (~90%)
- **Higher Recall** (~88%)

These metrics indicate it is more reliable at detecting **true positives** (successful startups) while minimizing false alarms.

### 3. **Scalability & Speed**

KNN becomes **computationally expensive** as:
- Dataset size increases
- Feature space expands

Logistic Regression trains and predicts **much faster**, even with over 100 features.

### 4. **Stability in High Dimensions**

High-dimensional data tends to degrade KNN performance due to the **curse of dimensionality**. Logistic Regression handles this well, especially with standardized features.

### 5. **Outlier Robustness**

Logistic Regression is less affected by **outliers and noisy features**, especially when regularization is applied (L2 by default in `sklearn`).

---

## 🧪 Evaluation Details

### Confusion Matrix (Logistic Regression)

```
                  Predicted
              | Failed | Success
    ----------|--------|--------
    Failed    |   27   |   6
    Success   |   7    |   55
```

- **True Negatives (TN)**: 27
- **False Positives (FP)**: 6
- **False Negatives (FN)**: 7
- **True Positives (TP)**: 55

### Classification Report:

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Failed    | 0.79      | 0.82   | 0.81     |
| Success   | 0.90      | 0.89   | 0.89     |
| **Macro Avg** | 0.85      | 0.85   | 0.85     |
| **Weighted Avg** | 0.86      | 0.86   | 0.86     |

---

## 📉 Limitations of KNN

- **Computational Cost**: Prediction requires computing distances to all training samples
- **No Training Phase**: Entire dataset must be stored for inference
- **Sensitive to Irrelevant Features**: Needs careful feature selection or dimensionality reduction
- **Not Interpretable**: Provides no explanation for predictions
- **Worse Recall**: Fails to detect many successful startups (False Negatives)

---

## 📦 Requirements

Install the following Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📁 Project Structure

```
.
├── data.csv                         # Startup dataset
├── startup_success_failure.ipynb   # Main notebook (with full analysis)
├── README.md                       # This file
```

---

## 🚀 How to Run

Open the Jupyter notebook:

```bash
jupyter notebook startup_success_failure.ipynb
```

Run all cells to:

- Explore and preprocess data
- Train and evaluate Logistic Regression and KNN
- Visualize results and compare models

---

## 🔚 Conclusion

After thorough evaluation, **Logistic Regression** outperformed **KNN** in almost every aspect—interpretability, accuracy, efficiency, and scalability. For this binary classification task on structured business data, Logistic Regression is the most appropriate choice.
