import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(eval_pred):
    """Compute accuracy and F1 for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")  # For binary classification
    return {"accuracy": acc, "f1": f1}

def main():
    # 1) Read the dataset
    input_csv = "data/processed_data/political_compass_articles.csv"
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    if "cleaned_content" not in df.columns or "social_label" not in df.columns:
        raise ValueError("CSV must have 'cleaned_content' and 'social_label' columns.")
    
    # 2) Split data
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["social_label"],
        random_state=42
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["social_label"],
        random_state=42
    )

    # 3) Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # 4) Initialize tokenizer & model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_content"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # Rename "social_label" -> "labels" 
    tokenized_datasets = tokenized_datasets.rename_column("social_label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 5) Model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir="models/social_axis",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 7) Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    # 8) Train the model
    trainer.train()

    # 9) Evaluate on test
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("Test Results (Social Axis):", test_results)

    # Classification report
    raw_preds, labels, _ = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(raw_preds, axis=-1)
    print("\nClassification Report (Social Axis):")
    print(classification_report(labels, preds, digits=4))

    # 10) Save model
    trainer.save_model("models/social_axis")
    tokenizer.save_pretrained("models/social_axis")
    print("✅ Social axis model saved to 'models/social_axis'")

if __name__ == "__main__":
    os.makedirs("models/social_axis", exist_ok=True)
    main()
