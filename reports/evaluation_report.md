# Sentiment Analysis System — Evaluation Report

## 1. Project Overview
The project classifies customer reviews into three sentiment categories: Positive, Negative, and Neutral.

## 2. Dataset
- Dataset size: 10,000 reviews
- Input feature: Review text
- Target variable: Sentiment
- Train-test splitting: Group-based split to reduce identical review text appearing in both sets.

## 3. Model Selection
Multiple combinations of text vectorization and classification algorithms were compared:
- Bag of Words + Logistic Regression
- TF-IDF + Logistic Regression
- Bag of Words + Multinomial Naive Bayes
- TF-IDF + Multinomial Naive Bayes

GridSearchCV was used to tune the TF-IDF + Logistic Regression pipeline with grouped cross-validation.

## 4. Evaluation Results
- Test samples: [Enter actual value]
- Test accuracy: [Enter actual value]
- Macro precision: [Enter actual value]
- Macro recall: [Enter actual value]
- Macro F1-score: [Enter actual value]
- Weighted F1-score: [Enter actual value]
- Best grouped cross-validation Macro F1-score: [Enter actual value]

## 5. Best Model Parameters
[Copy the actual best parameters from evaluation_metrics.json.]

## 6. Confusion Matrix
[Insert the actual confusion matrix from confusion_matrix.csv.]

Rows represent actual labels and columns represent predicted labels. The label order is Negative, Neutral, Positive.

## 7. Conclusion
The selected model was saved as a reusable scikit-learn pipeline and integrated into a Streamlit application for interactive sentiment prediction.

The final performance conclusions should be based on the recorded evaluation results.
