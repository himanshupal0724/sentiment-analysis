
# AI-Powered Sentiment Analysis System

## 1. Project Overview

This project uses Natural Language Processing (NLP)
and Machine Learning to classify customer reviews
into three sentiment categories:

- Positive
- Negative
- Neutral

The application provides a web interface built with
Streamlit.

## 2. Problem Statement

Manually analyzing large numbers of customer reviews
is time-consuming. This project aims to automate
sentiment classification using supervised machine
learning.

## 3. Dataset

- Total records: 10,000
- Input feature: review_text
- Target variable: sentiment
- Classes: Positive, Negative, Neutral

The dataset also includes review_id, category, and source.

## 4. Methodology

1. Data inspection and cleaning
2. Exploratory Data Analysis
3. Text preprocessing
4. Bag of Words and TF-IDF feature extraction
5. Model comparison
6. Grouped train-test splitting
7. Cross-validation and hyperparameter tuning
8. Model evaluation
9. Pipeline serialization
10. Streamlit deployment

## 5. Models

Baseline experiments:
- Bag of Words + Logistic Regression
- TF-IDF + Logistic Regression
- Bag of Words + Multinomial Naive Bayes
- TF-IDF + Multinomial Naive Bayes

Final tuned model: [Insert actual model and parameters]

## 6. Evaluation Results

Test accuracy: [Insert measured value]
Macro precision: [Insert measured value]
Macro recall: [Insert measured value]
Macro F1-score: [Insert measured value]

See reports/ for the complete classification report
and confusion matrix.

## 7. Project Structure

- notebooks/: analysis and model training
- models/: saved trained pipeline
- reports/: evaluation outputs
- app.py: Streamlit application
- text_preprocessing.py: shared text cleaning
- requirements.txt: dependencies

## 8. Installation

Create and activate a Python virtual environment.

Install dependencies:

    pip install -r requirements.txt

Run the application:

    python -m streamlit run app.py

## 9. Limitations

Repeated review texts occur in the dataset.
Grouped splitting reduces exact-text leakage, but
the dataset contains only 300 unique review texts.
Performance on genuinely new customer language
may therefore differ from the measured test results.

## 10. Future Improvements

- Evaluate on a larger dataset with more unique reviews
- Test on independently collected customer reviews
- Improve probability calibration
- Add more detailed model monitoring
