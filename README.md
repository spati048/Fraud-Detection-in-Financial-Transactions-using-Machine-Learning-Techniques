# Enhancing Fraud Detection in Financial Transactions Using Machine Learning  

## Overview  
This project applies **machine learning techniques in R** to detect fraudulent financial transactions. Using a dataset of ~1,000 transactions, we performed **EDA, feature engineering, and model development** with Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine (SVM).  

Our best model, **Logistic Regression**, achieved an **AUC of 0.92** and helped reduce **false positives by 25%**, improving accuracy and customer trust.  

## Dataset  
- **Name:** Fraud Detection Dataset  
- **Observations:** 1,010  
- **Features:** 19  
- **Target Variable:** Fraud (1 = Fraudulent, 0 = Legitimate)  
- **Key Features:**  
  - TransactionAmount  
  - IsForeignTransaction  
  - NumChargebacks  
  - HasEmailDomainBlacklisted  

## Methodology  
1. **Data Cleaning** – Handled missing values, duplicates, and type inconsistencies.  
2. **EDA** – Identified correlations, outliers, and anomalies using visualizations.  
3. **Feature Engineering** – Transformed categorical variables, scaled features, and selected predictors.  
4. **Model Development** – Trained and tuned Logistic Regression, Random Forest, Gradient Boosting (XGBoost), and SVM.  
5. **Evaluation** – Compared models using ROC curves, AUC, precision-recall metrics, and confusion matrices.  
6. **Deployment Simulation** – Tested real-time fraud flagging in a simulated environment.  

## Results  

| Model                | AUC  | Strengths                            | Weaknesses                     |
|-----------------------|------|--------------------------------------|--------------------------------|
| Logistic Regression   | 0.92 | High AUC, simple, interpretable      | Limited for non-linear data     |
| Random Forest         | 0.89 | Robust, handles outliers             | Slower for large datasets       |
| Gradient Boosting     | 0.88 | Captures complex patterns            | Computationally intensive       |
| Support Vector Machine| 0.78 | Handles high-dimensional data        | Low performance in this case    |  

**Key Insights:**  
- Logistic Regression outperformed others with the steepest ROC curve.  
- Random Forest & Gradient Boosting effectively captured complex fraud patterns.  
- SVM was less effective but highlighted challenges in high-dimensional data.  

## Real-World Applications  
- **Banking:** Real-time fraud detection & AML compliance.  
- **E-Commerce:** Preventing chargeback and fake account fraud.  
- **Insurance:** Detecting fraudulent claims.  
- **Government:** Preventing tax fraud and fund misuse.  
- **Healthcare & Retail:** Identifying billing and promotional fraud.  

## Implementation & Challenges  
- **Integration:** Fraud detection integrated into transaction systems.  
- **Real-Time Monitoring:** Dashboards to flag high-risk transactions.  
- **Challenges:**  
  - Imbalanced dataset → handled with **SMOTE**.  
  - Reducing false positives → optimized thresholds.  
  - Real-time constraints → algorithm tuning for lower latency.  

## References  
- Microsoft Learn. (2024). [Use R to detect fraud](https://learn.microsoft.com/en-us/fabric/data-science/r-fraud-detection)  
- ProjectPro. (2024). [Credit Card Fraud Detection Project](https://www.projectpro.io/article/credit-card-fraud-detection-project-with-source-code-in-python/568)  
- DataFlair. (n.d.). [Credit Card Fraud Detection](https://data-flair.training/blogs/credit-card-fraud-detection-python-machine-learning/)  
