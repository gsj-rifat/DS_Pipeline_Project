***Fashion Forward Forecasting - StyleSense Predictive Model***


This project involves building a predictive model to analyze product reviews from StyleSense, a rapidly growing online women's clothing retailer. The goal is to automate the process of predicting whether a customer would recommend a product based on their review text, customer age, product category, and other relevant features.

Project Overview
StyleSense has seen a large influx of new customers, which has resulted in a backlog of product reviews with missing data. The task is to leverage the available data, particularly the reviews with complete information, to build a machine learning model that can predict whether a customer would recommend a product.

This model will help StyleSense:

Gain valuable insights into customer satisfaction.
Identify trending products based on review analysis.
Provide a better shopping experience for their growing customer base.


*Key Features*

Data Handling: The dataset contains numerical, categorical, and textual features.
Text Processing: SpaCy is used for text preprocessing, including lemmatization and removal of stop words and punctuation.
Machine Learning Model: A Random Forest Classifier is used to predict the recommendation (0 or 1) based on various features.
Hyperparameter Tuning: Grid Search is applied to fine-tune the Random Forest model and find the optimal hyperparameters.
Model Evaluation: The model is evaluated using accuracy, precision, recall, F1-score, and other metrics.
Visualization: Various performance metrics, including confusion matrix, ROC curve, and grid search results, are visualized for better model understanding.

*Data*

The dataset reviews.csv includes the following columns:

Clothing ID (Exclusion in the model)
Age (Numerical)
Title (Textual)
Review Text (Textual)
Positive Feedback Count (Numerical)
Division Name (Categorical)
Department Name (Categorical)
Class Name (Categorical)
Recommended IND (Target: 0 or 1, binary)
The task involves predicting the Recommended IND based on the other features.

*Approach*
Data Preprocessing:

Remove unnecessary columns.
Handle missing values and text data.
Feature engineering, including scaling numerical data and encoding categorical data.
Model Training:

A Random Forest Classifier is used as the base model.
Tuning hyperparameters using Grid Search.

*Model Evaluation:*

Evaluate model performance using classification metrics (accuracy, precision, recall, F1-score).
Visualize the confusion matrix and ROC curve to understand classification performance.

Grid Search Visualization:

Visualize how hyperparameters like n_estimators and max_depth affect model performance.
Feature Importance:

Analyze feature importance to understand which features are driving model predictions.

*Libraries Used*

pandas: For data manipulation and analysis.
scikit-learn: For machine learning and model evaluation.
spaCy: For text preprocessing and natural language processing.
matplotlib & seaborn: For data visualization.
numpy: For numerical operations.
