import pandas as pd
import re
import nltk
import contractions
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("omw-1.4")
nltk.download("wordnet")

# Define stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans and preprocesses the given text."""
    if not isinstance(text, str):
        return ""

    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove emojis & special characters
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = text.encode("ascii", "ignore").decode()  # Remove non-ASCII characters

    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Replace with space to maintain spacing

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove single characters (isolated letters)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords and lemmatize words
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Rejoin words into a cleaned sentence
    cleaned_text = " ".join(cleaned_words)

    return cleaned_text

def process_dataset(input_file, output_file):
    """Loads dataset, cleans the text columns, and saves the updated data."""
    try:
        # Load CSV
        df = pd.read_csv(input_file)

        # Ensure required columns exist
        if "content" not in df.columns or "title" not in df.columns:
            raise ValueError("CSV must contain 'content' and 'title' columns.")

        # Clean 'content' and 'title' separately
        df["cleaned_content"] = df["content"].apply(clean_text)
        df["cleaned_title"] = df["title"].apply(clean_text)

        # Drop rows where 'cleaned_content' or 'cleaned_title' is empty
        df = df.dropna(subset=["cleaned_content", "cleaned_title"])

        # Save everything, including the new cleaned columns
        df.to_csv(output_file, index=False)
        print(f"✅ Cleaned data saved to {output_file}")

    except Exception as e:
        print(f"❌ Error processing dataset: {e}")

if __name__ == "__main__":
    input_csv = "data/raw_articles/scraped_articles.csv"
    output_csv = "data/processed_data/cleaned_articles.csv"
    
    process_dataset(input_csv, output_csv)
