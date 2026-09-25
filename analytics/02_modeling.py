import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load from committed CSV
df = pd.read_csv("titanic.csv")

# Select relevant features and target
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

X = df[features]
y = df[target]

# 2. Stratified Train/Test Split (Justification: preserves class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Leakage-Free Preprocessing via ColumnTransformer
numeric_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4. Train Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, oob_score=True, random_state=42
    ),
}

results = []

for name, clf in classifiers.items():
  pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
  pipeline.fit(X_train, y_train)

  y_pred = pipeline.predict(X_test)
  y_prob = pipeline.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_prob)

  results.append({
      "Model": name,
      "Accuracy": acc,
      "Precision": prec,
      "Recall": rec,
      "F1 Score": f1,
      "AUC": auc,
  })

metrics_df = pd.DataFrame(results)
print("\n--- Classifier Metrics Comparison ---")
print(metrics_df)

# 5. Save best performing complete pipeline using joblib
best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100, oob_score=True, random_state=42
            ),
        ),
    ]
)
best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, "model_pipeline.pkl")
print(
    "\nSaved complete pipeline to model_pipeline.pkl successfully!"
)

# Verify reload
loaded_pipeline = joblib.load("model_pipeline.pkl")
sample_prediction = loaded_pipeline.predict(X_test.head(1))
print("Test prediction on reloaded pipeline:", sample_prediction)