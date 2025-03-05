from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model
with open("../models/bias_model.pkl", "rb") as f:
    model = pickle.load(f)

vectorizer = TfidfVectorizer()

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return jsonify({"bias": prediction})

if __name__ == "__main__":
    app.run(debug=True)