import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_axis(text, tokenizer, model):
    """Returns 0 or 1 for the given axis."""
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = np.argmax(logits.numpy(), axis=-1)
    return pred[0]  # 0 or 1

def predict_political_compass(text):
    # Load your economic axis model
    econ_tokenizer, econ_model = load_model_and_tokenizer("models/economic_axis")
    # Load your social axis model
    soc_tokenizer, soc_model = load_model_and_tokenizer("models/social_axis")
    
    # Predict economic axis
    econ_label = predict_axis(text, econ_tokenizer, econ_model)
    # Predict social axis
    soc_label = predict_axis(text, soc_tokenizer, soc_model)
    
    # Combine results into quadrant
    if econ_label == 0 and soc_label == 0:
        quadrant = "Authoritarian Left"
    elif econ_label == 0 and soc_label == 1:
        quadrant = "Libertarian Left"
    elif econ_label == 1 and soc_label == 0:
        quadrant = "Authoritarian Right"
    else:
        quadrant = "Libertarian Right"
    
    return quadrant

if __name__ == "__main__":
    # Example usage
    sample_text = "Government must regulate businesses heavily for public welfare."
    pred = predict_political_compass(sample_text)
    print(f"Text: {sample_text}")
    print(f"Political Compass Prediction: {pred}")
