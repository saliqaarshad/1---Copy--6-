import streamlit as st
import joblib
import numpy as np

# Load model and vectorizers
model = joblib.load('spam_model.pkl')
vectorizer_message = joblib.load('vectorizer_message.pkl')
vectorizer_subject = joblib.load('vectorizer_subject.pkl')

st.title("📧 Email Spam Detection App")
st.write("Enter the subject and message to check if it's spam or ham.")

# User inputs
subject_input = st.text_input("Subject")
message_input = st.text_area("Message")

if st.button("Predict"):
    if subject_input.strip() == "" or message_input.strip() == "":
        st.warning("Please fill in both Subject and Message.")
    else:
        # Transform inputs
        X_m = vectorizer_message.transform([message_input]).toarray()
        X_s = vectorizer_subject.transform([subject_input]).toarray()

        # Combine
        X_input = np.hstack((X_m, X_s))

        # Predict
        prediction = model.predict(X_input)[0]

        if prediction == 1:
            st.error("🚨 This email is likely SPAM!")
        else:
            st.success("✅ This email is likely HAM (not spam).")
