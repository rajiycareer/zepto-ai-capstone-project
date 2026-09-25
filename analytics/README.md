# Module 2: Analytics & Predictive Modeling Pipeline

## Key Findings & Interpretations
- **Missing Value Handling:** Columns with <5% missing values (`embarked`) had their rows dropped. Columns with 5%–30% missing values (`age`) were imputed using median values. Columns with excessive missingness (`deck`, >77%) were dropped.
- **Correlation Insights:** The strongest absolute off-diagonal correlations were observed between `pclass` and `fare` (negative correlation, as higher class numbers correspond to lower ticket prices) and `sibsp` and `parch` (family size association).
- **Model Recommendation:** **Random Forest** is recommended for deployment due to its superior generalization, robust handling of non-linear feature interactions, and strong ROC-AUC performance.