import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/processed_data/cleaned_articles.csv")

# Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
Y = df["bias"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, Y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")
print(classification_report(Y_test, predictions))  # Show precision, recall, F1-score

# Save model and vectorizer
with open("models/bias_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved to 'models/' directory.")
