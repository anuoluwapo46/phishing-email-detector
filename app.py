import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Page setup
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")
st.title("🛡️ Phishing Email Detector")
st.write("Paste any email text below to check if it is phishing or safe.")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load and train model
@st.cache_resource
def train_model():
    file_id = "1ipL5E5tGsenFUG-3oW-4HbzgGzRqbbGc"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)
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
    # Get logistic regression coefficients for explanation
    lr_model = model = ensemble.estimators_[1]
    feature_names = tfidf.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    return ensemble, tfidf, feature_names, coefficients

# Show loading message
with st.spinner("Loading model... please wait..."):
    model, tfidf, feature_names, coefficients = train_model()

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

        # Show result
        if prediction == 1:
            st.error("🚨 PHISHING EMAIL DETECTED!")
        else:
            st.success("✅ SAFE EMAIL")

        st.write(f"**Safe Probability:** {probability[0]:.2%}")
        st.write(f"**Phishing Probability:** {probability[1]:.2%}")

        # Plain English Explanation
        st.markdown("---")
        st.markdown("### 🔍 Why did the model make this decision?")

        # Get words that actually appear in this specific email
        email_words = set(cleaned.split())
        word_impacts = []

        for word in email_words:
            if word in feature_names:
                idx = list(feature_names).index(word)
                impact = coefficients[idx]
                word_impacts.append((word, impact))

        # Sort by absolute impact
        word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        # Split into phishing and safe words
        phishing_words = [w for w, v in word_impacts if v > 0][:5]
        safe_words = [w for w, v in word_impacts if v < 0][:5]

        if prediction == 1:
            st.markdown("#### This email was flagged as **phishing** because:")
            if phishing_words:
                for word in phishing_words:
                    st.markdown(f"- 🚩 The word **'{word}'** is commonly found in phishing emails")
            if safe_words:
                st.markdown("#### These words slightly suggested it could be safe:")
                for word in safe_words:
                    st.markdown(f"- ✅ The word **'{word}'** is more common in safe emails")
            st.markdown(f"> ⚠️ Overall the suspicious words outweighed the safe ones, "
                       f"giving a **{probability[1]:.2%} phishing probability**.")
        else:
            st.markdown("#### This email was classified as **safe** because:")
            if safe_words:
                for word in safe_words:
                    st.markdown(f"- ✅ The word **'{word}'** is commonly found in safe emails")
            if phishing_words:
                st.markdown("#### These words slightly raised suspicion:")
                for word in phishing_words:
                    st.markdown(f"- 🚩 The word **'{word}'** is sometimes found in phishing emails")
            st.markdown(f"> ✅ Overall the safe words outweighed the suspicious ones, "
                       f"giving a **{probability[0]:.2%} safe probability**.")
