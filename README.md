# Spam Mail Detection

## Overview
This project detects whether an email is spam or not using machine learning. It classifies messages into spam (0) or ham (1) based on their content.

## Dataset
- Dataset used: spam.csv
- Total records: 5572
- Features:
  - v1 → Label (spam/ham)
  - v2 → Email text

## Project Workflow

### Data Preprocessing
- Removed unnecessary columns
- Handled missing values
- Converted labels:
  - spam → 0
  - ham → 1

### Feature Extraction
- Used TF-IDF Vectorizer to convert text into numerical features

### Data Splitting
- Training set: 80%
- Testing set: 20%

### Handling Imbalance
- Applied SMOTE to balance spam and ham data

### Model Building
- Logistic Regression model used for classification

## Results

### Testing Accuracy
- Accuracy: 98%

### Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-score

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Imbalanced-learn (SMOTE)

## Output
- Model saved as: model.pkl
- Vectorizer saved as: vectorizer.pkl
