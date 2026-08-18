# AI-Powered Phishing Email Detection System

## Overview
This project is a machine learning-based system that detects phishing emails by analyzing email content and suspicious URL patterns.

## Features
- Email text preprocessing
- Phishing probability prediction
- Confidence score generation
- Streamlit web interface

## Objectives
classify email text as phishing or legitimate;
apply NLP/text preprocessing and machine-learning techniques;
provide probability-based predictions;
provide an interactive web interface.

## Methodology
Email Input

     ↓
     
Text Cleaning

     ↓
     
TF-IDF Feature Extraction

     ↓
     
 ┌───────────────┐
 
 │ Random Forest     │
 
 │ Logistic Reg.     │
 
 │ Multinomial NB    │
 
 └───────────────┘
 
     ↓
     
Soft Voting Ensemble

     ↓
     
Prediction + Probability

     ↓
     
Plain-English Explanation

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Streamlit
- Numpy
- TF-IDF
- Random Forest
- Logistic Regression
- Multinominal Naive Bayes
  
## Sample Output
**Prediction:** Phishing  
**Confidence:** 96.4%

## Author
Kehinde Ayomide Emmanuel
Federal University Oye-Ekiti (FUOYE)# phishing-email-detector
