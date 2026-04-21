import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import shap

# Page setup
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")
st.title("🛡️ Phishing Email Detector")
st.write("Paste any email text below to check if it is phishing or safe.")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load and train model
@st.cache_resource
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/anuoluwapo46/phishing-email-detector/main/Phishing_Email.csv")
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(subset=['Email Text'])
    df = df.reset_index(drop=True)
    df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df['Cleaned Text'] = df['Email Text'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Cleaned Text'])
    y = df['Label']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    nb = MultinomialNB()
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('nb', nb)],
        voting='soft'
    )
    ensemble.fit(X, y)
    return ensemble, tfidf

# Show loading message
with st.spinner("Loading model... please wait..."):
    model, tfidf = train_model()

st.success("Model ready!")

# Email input box
email_input = st.text_area("📧 Paste Email Text Here:", height=200)

# Check button
if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please paste an email text first!")
    else:
        cleaned = clean_text(email_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 PHISHING EMAIL DETECTED!")
        else:
            st.success("✅ SAFE EMAIL")

        st.write(f"**Safe Probability:** {probability[0]:.2%}")
        st.write(f"**Phishing Probability:** {probability[1]:.2%}")

        # SHAP explanation
        st.markdown("### 🔍 SHAP Explanation")
        lr_model = model.estimators_[1]
        explainer = shap.LinearExplainer(
            lr_model, vectorized,
            feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(vectorized)
        feature_names = tfidf.get_feature_names_out()
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values, vectorized.toarray(),
            feature_names=feature_names,
            max_display=10, show=False
        )
        st.pyplot(fig)
