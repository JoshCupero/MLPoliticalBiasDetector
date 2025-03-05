import re
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")
print(" spaCy model loaded successfully!")

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\D+", " ", text)  #Remove numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = " ".join([token.lemma_ for token in nlp(text)])  # Lemmatize text
    return text

# Load scraped data
df = pd.read_csv("data/allsides_bias_data.csv")

# Apply cleaning
df["cleaned_text"] = df["source"].apply(clean_text)

# Save cleaned data
df.to_csv("data/cleaned_news_data.csv", index=False)
print("C;eaned data saved to data/cleaned_news_data.csv")