import pandas as pd
import numpy as np
import streamlit as st
import base64  # Added for background image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Background Image Function ---
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background for all users
set_background("image-removebg-preview (9).png")  # Updated with your image file name

# Load dataset
df = pd.read_csv("Hamspam.csv", encoding="ISO-8859-1")

# Preprocessing
df.columns = ['label', 'message']  # Renaming columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Streamlit UI
st.title("\U0001F4E7 Spam Detector")
st.write("Enter an email or message to check if it's spam or ham.")

user_input = st.text_area("Enter message:", height=150)
if st.button("Predict"):
    if user_input:
        transformed_message = vectorizer.transform([user_input])
        prediction = model.predict(transformed_message)
        if prediction[0] == 1:
            st.markdown("<p style='color:red; font-size:20px;'>\U0001F6A8 Spam Email Detected!</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:green; font-size:20px;'>\u2705 This is a Ham Email.</p>", unsafe_allow_html=True)
    else:
        st.write("Please enter a message.")
