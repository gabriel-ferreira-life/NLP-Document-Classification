## Config
# Import Libraries
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import evaluate
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split, cross_val_score

# Load the config file
with open('../config/config.json', 'r') as f:
    config = json.load(f)

file_path = config["data_loc"]

# Define file path
file_name = "QTL_text.json"
final_path = os.path.join(file_path, file_name) 


## Load dataset
# Load json file
df = pd.read_json(final_path)
df = df.drop(columns=['Journal'])
print(f"Shape of the original dataset: {df.shape}", "\n")

## Pre-process
# Define predictor and target features
X = df.drop(columns=['Category'])
y = df['Category']

# Split train and test
X_train_corpus, X_test, y_train_corpus, y_test = train_test_split(X,y, test_size=.2, random_state=42, stratify=y)

# Split train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train_corpus,y_train_corpus, test_size=.2, random_state=42, stratify=y_train_corpus)

## Load Pre-trained Model
# Define pre-trained model path
model_path = "google-bert/bert-base-uncased"

# Load model tokeninzer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with binary classification head
id2label = {0: "Not Related", 1: "Related"}
label2id = {"Not Related": 0, "Related": 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id,)

# Freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad=False

# Unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad=True


# Training Data
train_data = {"text": X_train['Title'], "labels": y_train}
train_dataset = Dataset.from_dict(train_data)

# Validation Data
val_data = {"text": X_val['Title'], "labels": y_val}
val_dataset = Dataset.from_dict(val_data)

# Test Data
test_data = {"text": X_test['Title'], "labels": y_test}
test_dataset = Dataset.from_dict(test_data)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Define text preprocessing
def preprocess_function(examples):
    # Return tokenized text with truncation
    return tokenizer(
        examples['text'], 
        truncation=True) # Truncate abstracts greater than 512 tokens

# Preprocess all datasets
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Uniform sample lenght

# Load metrics
f1_score = evaluate.load("f1", config="macro")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    # Get predictions
    predictions, labels = eval_pred

    # Apply softmax to get probabilities
    probabilities = np.exp(predictions)  / np.exp(predictions).sum(-1, keepdims=True)

    # Use probabilities of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]

    # Compute AUC
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)


    # Predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)

    # Compute Accuracy
    f1 = np.round(f1_score.compute(predictions=predicted_classes, references=labels)['f1'], 4)

    return {"F1": f1, "AUC": auc}

# Hyperparameters
lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir="experiment_outputs",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# trainer.train()
# To resume from the last checkpoint in your output_dir:
trainer.train(resume_from_checkpoint=True)

