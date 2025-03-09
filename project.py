# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load the English language model in spacy
nlp = spacy.load("en_core_web_sm")

# 1. Load the dataset
file_path = Path(r"C:\Users\nn\PycharmProjects\DS_Pipeline_Project\starter\data\reviews.csv")
data = pd.read_csv(file_path)

# 2. Data Preprocessing

# Remove the 'Clothing ID' column as instructed
data = data.drop(columns=['Clothing ID'])

# Handle missing values by removing rows with missing target or review text
data = data.dropna(subset=['Review Text', 'Recommended IND'])

# Check for any remaining missing values
print("Missing values:\n", data.isnull().sum())

# 3. Feature Engineering

# Split data into features (X) and target (y)
X = data.drop(columns=['Recommended IND'])
y = data['Recommended IND']

# Convert categorical columns to numerical using Label Encoding (for Division, Department, Class)
categorical_cols = ['Division Name', 'Department Name', 'Class Name']
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Normalize the 'Age' and 'Positive Feedback Count' columns
scaler = StandardScaler()
X[['Age', 'Positive Feedback Count']] = scaler.fit_transform(X[['Age', 'Positive Feedback Count']])

# 4. Text Processing with SpaCy
def process_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Apply text processing (lemmatization and stopword removal) to the 'Review Text'
X['Processed Review Text'] = X['Review Text'].apply(process_text)

# 5. Feature Extraction from Text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(X['Processed Review Text'])

# Combine text features with other features
X_combined = hstack([X.drop(columns=['Review Text', 'Processed Review Text']), X_text])

# 6. Train/Test Split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 7. Model Training and Pipeline Creation

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline for text processing + model training
pipeline = make_pipeline(model)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# 8. Model Evaluation and Performance Metrics

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')

# Plot ROC Curve
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

# 9. Hyperparameter Tuning (Optional)

# Hyperparameter tuning for Random Forest
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [10, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# 10. Feature Importance Visualization

# Get feature importance from the best model
importances = model.feature_importances_

# Sort feature importance
indices = np.argsort(importances)[::-1]

# Plot the top 10 important features
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [f'Feature {i+1}' for i in range(10)], rotation=90)
plt.xlim([-1, 10])
plt.show()

