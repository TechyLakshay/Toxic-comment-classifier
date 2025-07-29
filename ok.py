import os
os.environ["USE_TF"] = "0"  # 👈 Force to use only PyTorch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
import torch

# Load dataset
dataset = load_dataset("llangnickel/long-covid-classification-data", download_mode="force_redownload")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Rename label column to match Transformers' expected format
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Set dataset format for PyTorch
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Metric function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Training arguments
training_args = TrainingArguments(
    output_dir="./long_covid_sentiment_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model
trainer.evaluate()

# Save model and tokenizer
model.save_pretrained("./long_covid_sentiment_model")
tokenizer.save_pretrained("./long_covid_sentiment_model")

# Example inference
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="./long_covid_sentiment_model",
    tokenizer="./long_covid_sentiment_model",
    framework="pt",  # 👈 Force PyTorch
    device=0 if torch.cuda.is_available() else -1
)

result = classifier("This document discusses persistent Long COVID symptoms.")
print(result)
