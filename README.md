
# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning. We implemented and compared four different algorithms: Decision Tree, Logistic Regression, Random Forest, and XGBoost. The goal is to identify the most effective model for fraud detection based on performance metrics.

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description:** The dataset contains transactions made by European cardholders in September 2013. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
- **Features:** 30 features (V1-V28 are PCA components, plus 'Time', 'Amount', and 'Class' where 'Class' is the target variable: 1 for fraud, 0 for non-fraud).

## Models Implemented
1. **Decision Tree**  
  Implemented in `Decision_Tree_Fraud_Detection.ipynb`  
  Simple, interpretable model for classification tasks.

2. **Logistic Regression**  
  Implemented in `logistic_regression.ipynb`  
  A linear model suitable for binary classification.

3. **Random Forest**  
  Implemented in `random forest.ipynb`  
  An ensemble of decision trees to improve accuracy and control overfitting.

4. **XGBoost**  
  Implemented in `xgboost_only.ipynb`  
  A powerful gradient boosting algorithm known for high performance on tabular data.

## Model Comparison
- The models were evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Due to the imbalanced nature of the dataset, special attention was given to Recall and ROC-AUC to ensure effective fraud detection.
- Results and comparison plots are available in `model_comparison.ipynb`.

### Results Summary
| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
|--------------------|----------|-----------|--------|----------|---------|---------------|
| Decision Tree      | 0.9694   | 0.0474    | 0.8776 | 0.0899   | 0.9166  | 0.4498        |
| Logistic Regression| 0.9441   | 0.0276    | 0.9184 | 0.0535   | 0.9709  | 0.7226        |
| Random Forest      | 0.9986   | 0.5621    | 0.8776 | 0.6853   | 0.9676  | 0.8757        |
| XGBoost            | 0.9972   | 0.3668    | 0.8571 | 0.5138   | 0.9783  | 0.8467        |

**Best Model:** Random Forest with F1-Score of 0.6853

*Note: The comparison notebook now uses the same preprocessing, parameters, and thresholds as the individual model notebooks to ensure consistent results.*

## Model Improvements
After initial evaluation, we implemented the following optimizations:

1. **XGBoost Hyperparameter Tuning**
   - Used RandomizedSearchCV to optimize parameters
   - Best parameters: max_depth=6, n_estimators=300, learning_rate=0.2, colsample_bytree=0.9
   - F1-Score improved from 0.7089 to 0.7713 (+8.8%)

2. **Logistic Regression Threshold Optimization**
   - Optimized decision threshold from 0.5 to 0.85
   - F1-Score improved from 0.1088 to 0.2812 (+158.5%)
   - Significantly better precision while maintaining high recall

## How to Run
1. Clone this repository.
2. Install required Python packages (see below).
3. Open the notebooks in Jupyter or VS Code and run the cells in order.

### Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## References
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)