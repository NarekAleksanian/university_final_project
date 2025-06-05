
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

## 📐 Mathematical Formulation of Logistic Regression

Logistic Regression aims to model the probability that a binary outcome $y \in \{0,1\}$ occurs given input features $\mathbf{x} \in \mathbb{R}^n$. We denote the parameter vector by $oldsymbol{	heta} \in \mathbb{R}^n$ and bias (intercept) by $	heta_0$.

1. **Linear Combination (Logit)**  
   For a given feature vector $\mathbf{x}$, compute the linear combination:
   $$
   z = 	heta_0 + \sum_{j=1}^{n} 	heta_j x_j = oldsymbol{	heta}^\mathrm{T} \mathbf{x} + 	heta_0.
   $$

2. **Sigmoid (Logistic) Function**  
   The logistic function $\sigma(z)$ maps $z$ to a probability in $(0,1)$:
   $$
   \sigma(z) \;=\; rac{1}{1 + e^{-z}}.
   $$
   Therefore, the hypothesis (predicted probability) is:
   $$
   h_{oldsymbol{	heta}}(\mathbf{x}) \;=\; \sigma(oldsymbol{	heta}^\mathrm{T} \mathbf{x} + 	heta_0).
   $$

3. **Probability Interpretation**  
   We interpret:
   $$
   P(y=1 \mid \mathbf{x};\, oldsymbol{	heta}) \;=\; h_{oldsymbol{	heta}}(\mathbf{x}), 
   \quad
   P(y=0 \mid \mathbf{x};\, oldsymbol{	heta}) \;=\; 1 - h_{oldsymbol{	heta}}(\mathbf{x}).
   $$

4. **Cost Function (Log-Likelihood / Cross-Entropy)**  
   For $m$ training examples $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, the **log-likelihood** for logistic regression is:
   $$
   \ell(oldsymbol{	heta}) \;=\; \sum_{i=1}^{m} \Big[
     y^{(i)} \log ig(h_{oldsymbol{	heta}}(\mathbf{x}^{(i)})ig) \;+\; 
     ig(1 - y^{(i)}ig) \log ig(1 - h_{oldsymbol{	heta}}(\mathbf{x}^{(i)})ig)
   \Big].
   $$
   We typically minimize the **negative log-likelihood**, known as the **logistic cost** or **cross-entropy loss**:
   $$
   J(oldsymbol{	heta}) \;=\; -\,rac{1}{m}
   \sum_{i=1}^{m} \Big[
     y^{(i)} \log ig(h_{oldsymbol{	heta}}(\mathbf{x}^{(i)})ig) \;+\; 
     ig(1 - y^{(i)}ig) \log ig(1 - h_{oldsymbol{	heta}}(\mathbf{x}^{(i)})ig)
   \Big].
   $$

5. **Gradient Computation**  
   To minimize $J(oldsymbol{	heta})$ via gradient descent (or one of its variants), we compute the partial derivative with respect to each parameter $	heta_j$:
   $$
   rac{\partial J(oldsymbol{	heta})}{\partial 	heta_j}
   \;=\; rac{1}{m} \sum_{i=1}^{m} ig( h_{oldsymbol{	heta}}(\mathbf{x}^{(i)}) - y^{(i)} ig)\, x_j^{(i)},
   \quad 
   	ext{for } j = 0,1,\dots,n,
   $$
   noting that $x_0^{(i)} = 1$ for the intercept term.

6. **Parameter Update Rule**  
   With a learning rate $lpha$, gradient descent updates each parameter as:
   $$
   	heta_j := 	heta_j - lpha \,rac{\partial J(oldsymbol{	heta})}{\partial 	heta_j}, 
   \quad 
   orall\, j = 0,1,\dots,n.
   $$

7. **Decision Boundary**  
   After training, we predict class labels by thresholding the probability at $0.5$:
   $$
   \hat{y} = 
   egin{cases}
     1, & 	ext{if } h_{oldsymbol{	heta}}(\mathbf{x}) \ge 0.5,\
     0, & 	ext{otherwise.}
   \end{cases}
   $$

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
