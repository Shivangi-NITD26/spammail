**Spam Mail Detection Model**

**Overview**

This machine learning project aims to classify SMS messages as spam or ham (non-spam) using Natural Language Processing (NLP) and Logistic Regression. The model helps filter unwanted spam messages automatically and improves communication security.

**Dataset Overview**

Source: Kaggle – SMS Spam Collection Dataset  
(https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

- Records: 5572 SMS messages
- Columns: 5
- Target Variable:
  - spam = 0
  - ham = 1

**Data Preprocessing**

- Removed null values
- Replaced missing values with empty strings
- Encoded labels:
  - spam → 0
  - ham → 1
- Selected message column (`v2`) as input feature
- Split dataset into training and testing sets

**Feature Extraction**

Applied TF-IDF Vectorization to convert text messages into numerical feature vectors.

**Handling Imbalanced Data**

Used SMOTE (Synthetic Minority Over-sampling Technique) to balance spam and ham classes in the training dataset.

**Model Building**

- Logistic Regression Classifier

**Model Training**

- Trained using TF-IDF transformed SMS data
- Applied SMOTE-balanced training dataset

**Results**

Training Accuracy: 99.25%  
Testing Accuracy: 98.11%

**Performance Metrics**

- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

**Technologies Used**

- Python
- NumPy
- Pandas
- Scikit-learn
- NLP (TF-IDF Vectorizer)
- Imbalanced-learn (SMOTE)

**Output**

Saved trained files:
- model.pkl
- vectorizer.pkl

**Project Workflow**

1. Load SMS spam dataset
2. Preprocess and clean text data
3. Convert text into TF-IDF vectors
4. Apply SMOTE for balancing classes
5. Train Logistic Regression model
6. Evaluate model performance
7. Save trained model for future prediction
