import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore", category=FutureWarning)

# 🔹 More Extreme Reference Texts for Economic Axis
ECON_LEFT_REFERENCES = [
    "The government must ensure wealth redistribution to create fairness.",
    "High corporate taxes are necessary to fund public infrastructure and social programs.",
    "We need strong regulations to limit the power of large corporations.",
    "Universal basic income should be implemented for all citizens."
]
ECON_RIGHT_REFERENCES = [
    "Low taxes and minimal government intervention will allow the free market to thrive.",
    "Deregulation is essential for economic growth and personal freedom.",
    "Government involvement in the economy should be minimized.",
    "The private sector should handle healthcare, not the government."
]

# 🔹 More Extreme Reference Texts for Social Axis
SOC_AUTH_REFERENCES = [
    "The government must strictly regulate speech to prevent misinformation.",
    "National security is more important than individual privacy.",
    "A strong central government is needed to maintain order in society.",
    "Laws must enforce traditional moral values."
]
SOC_LIB_REFERENCES = [
    "Freedom of speech should have no restrictions, even for controversial opinions.",
    "Government surveillance is a violation of fundamental liberties.",
    "The state should not interfere with individual choices.",
    "Laws should not restrict personal behavior unless they harm others."
]

def main():
    input_csv = "data/processed_data/cleaned_articles.csv"
    output_csv = "data/processed_data/auto_labeled_political_compass.csv"
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found at {input_csv}")

    df = pd.read_csv(input_csv)
    if "cleaned_content" not in df.columns:
        raise ValueError("CSV must contain a 'cleaned_content' column.")

    # 🔹 Load Sentence-BERT Model
    model_name = "all-MiniLM-L12-v2"
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # 🔹 Compute Embeddings for All Articles
    print("Embedding all articles...")
    article_texts = df["cleaned_content"].fillna("").tolist()
    article_embeddings = model.encode(article_texts, show_progress_bar=True)

    # 🔹 Compute Centroids for Economic and Social Axes
    econ_left_embs = model.encode(ECON_LEFT_REFERENCES)
    econ_right_embs = model.encode(ECON_RIGHT_REFERENCES)
    soc_auth_embs = model.encode(SOC_AUTH_REFERENCES)
    soc_lib_embs = model.encode(SOC_LIB_REFERENCES)

    centroid_econ_left = np.mean(econ_left_embs, axis=0)
    centroid_econ_right = np.mean(econ_right_embs, axis=0)
    centroid_soc_auth = np.mean(soc_auth_embs, axis=0)
    centroid_soc_lib = np.mean(soc_lib_embs, axis=0)

    # 🔹 Define Cosine Similarity Function
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # 🔹 Assign Economic Axis (Left vs. Right)
    economic_labels = []
    economic_confidence_scores = []
    raw_econ_scores = []

    for emb in article_embeddings:
        sim_left = cosine_similarity(emb, centroid_econ_left)
        sim_right = cosine_similarity(emb, centroid_econ_right)
        raw_econ_scores.append(sim_left - sim_right)

    # Normalize Economic Scores
    norm_econ_scores = zscore(raw_econ_scores)

    for score in norm_econ_scores:
        if score >= 1.5:
            econ_label = "Far Left"
        elif score >= 0.5:
            econ_label = "Left"
        elif score <= -1.5:
            econ_label = "Far Right"
        elif score <= -0.5:
            econ_label = "Right"
        else:
            econ_label = "Moderate"

        economic_labels.append(econ_label)
        economic_confidence_scores.append(abs(score) * 100)  # Convert Z-score to percentage

    df["auto_economic_label"] = economic_labels
    df["economic_confidence"] = economic_confidence_scores

    # 🔹 Assign Social Axis (Authoritarian vs. Libertarian)
    social_labels = []
    social_confidence_scores = []
    raw_soc_scores = []

    for emb in article_embeddings:
        sim_auth = cosine_similarity(emb, centroid_soc_auth)
        sim_lib = cosine_similarity(emb, centroid_soc_lib)
        raw_soc_scores.append(sim_auth - sim_lib)

    # Normalize Social Scores
    norm_soc_scores = zscore(raw_soc_scores)

    for score in norm_soc_scores:
        if score >= 1.5:
            soc_label = "Far Authoritarian"
        elif score >= 0.5:
            soc_label = "Authoritarian"
        elif score <= -1.5:
            soc_label = "Far Libertarian"
        elif score <= -0.5:
            soc_label = "Libertarian"
        else:
            soc_label = "Moderate"

        social_labels.append(soc_label)
        social_confidence_scores.append(abs(score) * 100)  # Convert Z-score to percentage

    df["auto_social_label"] = social_labels
    df["social_confidence"] = social_confidence_scores

    # 🔹 Assign Final Quadrant Label
    df["auto_quadrant"] = df["auto_social_label"] + " " + df["auto_economic_label"]

    # 🔹 Save Auto-Labeled Dataset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Auto-labeling complete! Saved to {output_csv}")

if __name__ == "__main__":
    main()

# Load the labeled dataset
df = pd.read_csv("data/processed_data/auto_labeled_political_compass.csv")

# Show distribution of economic labels
print("Economic Label Distribution:")
print(df["auto_economic_label"].value_counts(), "\n")

# Show distribution of social labels
print("Social Label Distribution:")
print(df["auto_social_label"].value_counts(), "\n")

# Show distribution of final quadrants
print("Quadrant Label Distribution:")
print(df["auto_quadrant"].value_counts())

# Economic Axis Distribution
df["auto_economic_label"].value_counts().sort_index().plot(kind="bar", title="Economic Axis Distribution")
plt.xlabel("Economic Label")
plt.ylabel("Number of Articles")
plt.show()

# Social Axis Distribution
df["auto_social_label"].value_counts().sort_index().plot(kind="bar", title="Social Axis Distribution")
plt.xlabel("Social Label")
plt.ylabel("Number of Articles")
plt.show()

# Quadrant Distribution
df["auto_quadrant"].value_counts().sort_index().plot(kind="bar", title="Quadrant Distribution")
plt.xlabel("Quadrant")
plt.ylabel("Number of Articles")
plt.xticks(rotation=90)  # Rotate for readability
plt.show()