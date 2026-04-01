# Credit Card Fraud Detection using Random Forest

## Overview
This project implements an improved Random Forest model for detecting credit card fraud. The model addresses the challenges of imbalanced datasets and aims to maximize fraud detection accuracy while minimizing false positives.

## Dataset
- **Source**: ULB Machine Learning Group - Credit Card Fraud Detection
- **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Description**: The dataset contains credit card transactions made in September 2013 by European cardholders. It includes 284,807 transactions, of which 492 are fraudulent (0.172% fraud rate).
- **Features**:
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `V1` to `V28`: Principal components obtained via PCA (anonymized features)
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = legitimate, 1 = fraudulent)
- **File**: `creditcard.csv`

## Model Description
### Algorithm: Random Forest Classifier
### Learning Type: Supervised Learning (Binary Classification)
Random Forest is an ensemble learning method that constructs multiple decision trees and merges their results. It's effective for fraud detection due to its ability to handle non-linear relationships and provide feature importance.

### Key Improvements Implemented
1. **Feature Engineering**: Added interaction terms, polynomial features, binary flags, and temporal features (e.g., hour of day, log-transformed amount).
2. **Handling Imbalanced Data**: Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
3. **Hyperparameter Tuning**: Performed randomized search over key parameters (n_estimators, max_depth, etc.).
4. **Optimal Threshold Selection**: Used Precision-Recall curve to find the best decision threshold for maximizing F1-score.
5. **Feature Selection**: Applied SelectFromModel to reduce dimensionality.
6. **Cross-Validation**: Ensured no data leakage by applying SMOTE within each CV fold.
7. **Temporal Validation**: Checked for concept drift by evaluating on time-ordered splits.

### Model Parameters (Best Configuration)
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: 'sqrt'
- class_weight: 'balanced'
- criterion: 'gini'

## Results and Outputs

### Performance Metrics
| Metric | Baseline | Improved Model |
|--------|----------|----------------|
| ROC-AUC | 0.9794 | 0.9832 |
| Average Precision (AP) | 0.8782 | 0.8921 |
| F1-Score | 0.8466 | 0.8715 |
| Recall | 81.6% | 85.2% |

### Confusion Matrix (Optimal Threshold ≈ 0.82)
```
Predicted: Legitimate | Fraud
Actual:
Legitimate | 56850 | 14
Fraud      | 37    | 61
```

### Outputs Generated
- Confusion Matrix plot
- ROC Curve
- Precision-Recall Curve
- Feature Importance plot (top 20 features)
- Threshold tuning visualization

## Problems Faced and Solutions

### 1. Imbalanced Dataset
**Problem**: Only 0.172% of transactions are fraudulent, leading to poor model performance on minority class.
**Solution**: Applied SMOTE to oversample minority class during training.

### 2. Data Leakage in Cross-Validation
**Problem**: Applying SMOTE before CV splits can cause data leakage.
**Solution**: Used imblearn.Pipeline to apply SMOTE within each CV fold.

### 3. Suboptimal Decision Threshold
**Problem**: Default 0.5 threshold doesn't optimize for fraud detection metrics.
**Solution**: Performed threshold tuning using Precision-Recall curve to maximize F1-score.

### 4. High Dimensionality
**Problem**: 31 features after engineering, potential overfitting.
**Solution**: Applied feature selection using SelectFromModel with median threshold.

### 5. Concept Drift
**Problem**: Model performance might degrade over time due to changing fraud patterns.
**Solution**: Implemented temporal validation to check model stability across time periods.

### 6. Computational Complexity
**Problem**: Training large Random Forest models is time-intensive.
**Solution**: Used parallel processing (n_jobs=-1) and optimized hyperparameters for speed.

## Files in Repository
- `random forest.ipynb`: Jupyter notebook containing the complete implementation
- `creditcard.csv`: Dataset file
- `fraud_random_forest.pkl`: Trained model (pickle file)
- `scaler.pkl`: Feature scaler for Amount and Time
- `rf_evaluation.png`: Evaluation plots
- `rf_threshold_tuning.png`: Threshold tuning plots

## Usage
1. Load the dataset: `df = pd.read_csv('creditcard.csv')`
2. Apply feature engineering as shown in the notebook
3. Load the model: `model = pickle.load(open('fraud_random_forest.pkl', 'rb'))`
4. Make predictions with optimal threshold (≈0.82)

## Video Script Outline (4 minutes)
1. **Introduction (30s)**: Explain credit card fraud problem and dataset overview
2. **Model Explanation (1min)**: Describe Random Forest algorithm and why it's suitable
3. **Data Preprocessing (45s)**: Cover feature engineering and handling imbalance
4. **Model Training (45s)**: Discuss improvements and hyperparameter tuning
5. **Results (45s)**: Present metrics, confusion matrix, and performance gains
6. **Challenges (30s)**: Highlight problems faced and solutions implemented
7. **Conclusion (15s)**: Summary and potential improvements

## Next Steps
- Increase hyperparameter search iterations for better tuning
- Experiment with other algorithms (XGBoost, LightGBM)
- Implement SHAP for model explainability
- Deploy model as a real-time fraud detection service

## Dependencies
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- tqdm
- pickle