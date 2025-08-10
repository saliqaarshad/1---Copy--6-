import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = pd.read_csv('enron_spam_data.csv')
print('Data Loaded successfully')
print('Processing Data....')

# Drop missing rows
data.dropna(inplace=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing
data['Message'] = data['Message'].apply(preprocess_text)
data['Subject'] = data['Subject'].apply(preprocess_text)

# Use one vectorizer for consistency
vectorizer_message = TfidfVectorizer(stop_words='english', max_features=1000)
vectorizer_subject = TfidfVectorizer(stop_words='english', max_features=1000)

X_m = vectorizer_message.fit_transform(data['Message']).toarray()
X_s = vectorizer_subject.fit_transform(data['Subject']).toarray()

# Combine message + subject
X_message_subj = np.hstack((X_m, X_s))
y = data['Spam/Ham'].apply(lambda x: 1 if x == 'spam' else 0).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_message_subj, y, test_size=0.3, random_state=42)

# Train model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Evaluate
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Confusion Matrix:\n{cm}")

# Save model and vectorizers
joblib.dump(dt, 'spam_model.pkl')
joblib.dump(vectorizer_message, 'vectorizer_message.pkl')
joblib.dump(vectorizer_subject, 'vectorizer_subject.pkl')

print("Model and vectorizers saved successfully.")
