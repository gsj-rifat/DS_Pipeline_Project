import pandas as pd
import spacy
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load Data
file_path = Path(r"C:\Users\nn\PycharmProjects\DS_Pipeline_Project\starter\data\reviews.csv")
df = pd.read_csv(file_path)

# Drop Unnecessary Columns
df = df.drop(columns=["Clothing ID"])

# Define Features and Target
X = df.drop(columns=["Recommended IND"])
y = df["Recommended IND"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Preprocessing with SpaCy
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if pd.isnull(text):
        return ""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

X_train["Review Text"] = X_train["Review Text"].apply(clean_text)
X_test["Review Text"] = X_test["Review Text"].apply(clean_text)

# Feature Engineering Pipeline
numeric_features = ["Age", "Positive Feedback Count"]
numeric_transformer = StandardScaler()

categorical_features = ["Division Name", "Department Name", "Class Name"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

text_features = "Review Text"
text_transformer = TfidfVectorizer(max_features=5000)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("text", text_transformer, text_features),
    ]
)

# Model Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 1. Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 2. ROC Curve Visualization
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Hyperparameter Tuning
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model Evaluation
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# 3. Grid Search Results Visualization
# Extract results from GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

# Plot Grid Search results for n_estimators vs max_depth
plt.figure(figsize=(10, 6))
pivot_grid = results.pivot(index='param_classifier__max_depth', columns='param_classifier__n_estimators', values='mean_test_score')
sns.heatmap(pivot_grid, annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Grid Search Results: Mean Test Score vs. Hyperparameters')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.show()

# 4. Feature Importances Visualization
# Get feature importance from the best model
best_rf_model = best_model.named_steps['classifier']
importances = best_rf_model.feature_importances_

# Sort feature importance
indices = np.argsort(importances)[::-1]

# Plot the top 10 important features
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Top 10)")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [f'Feature {i+1}' for i in range(10)], rotation=90)
plt.xlim([-1, 10])
plt.show()
