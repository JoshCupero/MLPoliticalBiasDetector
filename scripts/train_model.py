import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load cleaned data
df = pd.read_csv("data/cleaned_news_data.csv")

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X  = vectorizer.fit_transform(df["cleaned_text"])
Y = df["bias"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
with open("models/bias_model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model saved to models/bias_model.pkl")   
