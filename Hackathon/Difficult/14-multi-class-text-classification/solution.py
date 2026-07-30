#!/usr/bin/env python3
"""Reference solution: Multi-Class Text Classification with Interpretability
Two approaches: (1) TF-IDF + LinearSVC + LIME, (2) DistilBERT + attention visualization
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not installed. Install with: pip install lime")

try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not installed. pip install transformers torch")

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/text_documents.csv")
X = df["text"]
y = df["category"]
categories = sorted(y.unique())
label_map = {c: i for i, c in enumerate(categories)}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Categories: {categories}")

# ================================
# Approach 1: TF-IDF + LinearSVC + LIME
# ================================
print("\n" + "=" * 60)
print("APPROACH 1: TF-IDF + LinearSVC + LIME")
print("=" * 60)

t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svc = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
svc.fit(X_train_vec, y_train)
train_time_1 = time.time() - t0

t0 = time.time()
y_pred_1 = svc.predict(X_test_vec)
inference_time_1 = time.time() - t0

acc_1 = accuracy_score(y_test, y_pred_1)
f1_1 = f1_score(y_test, y_pred_1, average="macro")
print(f"Accuracy: {acc_1:.4f}")
print(f"Macro F1: {f1_1:.4f}")
print(f"Training time: {train_time_1:.2f}s")
print(f"Inference time (test set): {inference_time_1:.4f}s")
print(classification_report(y_test, y_pred_1))

# --- LIME Explanations ---
if LIME_AVAILABLE:
    print("\n--- LIME Explanations (Approach 1) ---")
    # Need a predict_proba interface; wrap SVC with calibration
    calibrated = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42),
        cv=3, method="sigmoid"
    )
    calibrated.fit(X_train_vec, y_train)
    
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=categories, bow=False
    )
    
    def predict_proba_proxy(texts):
        vecs = vectorizer.transform(texts)
        return calibrated.predict_proba(vecs)
    
    # Explain a few test samples
    for idx in [0, 5, 12, 25, 40]:
        text = X_test.iloc[idx]
        true_label = y_test.iloc[idx]
        pred_label = calibrated.predict(vectorizer.transform([text]))[0]
        exp = explainer.explain_instance(
            text, predict_proba_proxy, num_features=10, top_labels=3
        )
        print(f"\nSample {idx}:")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print(f"  Top words for '{pred_label}':")
        for word, weight in exp.as_list(label=categories.index(pred_label)):
            print(f"    {word}: {weight:.4f}")

# ================================
# Approach 2: DistilBERT (if available)
# ================================
if TRANSFORMERS_AVAILABLE:
    print("\n" + "=" * 60)
    print("APPROACH 2: DistilBERT + Attention Visualization")
    print("=" * 60)
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(categories)
    )
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts.tolist()
            self.labels = labels.tolist() if labels is not None else None
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx], truncation=True, padding="max_length",
                max_length=128, return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(label_map[self.labels[idx]], dtype=torch.long)
            return item
    
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    
    training_args = TrainingArguments(
        output_dir="./bert_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="./bert_logs",
        logging_steps=50,
        report_to="none",
        disable_tqdm=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    t0 = time.time()
    trainer.train()
    train_time_2 = time.time() - t0
    
    t0 = time.time()
    preds = trainer.predict(test_dataset)
    inference_time_2 = time.time() - t0
    
    y_pred_2 = preds.predictions.argmax(axis=1)
    y_pred_2_labels = [categories[i] for i in y_pred_2]
    acc_2 = accuracy_score(y_test, y_pred_2_labels)
    f1_2 = f1_score(y_test, y_pred_2_labels, average="macro")
    print(f"Accuracy: {acc_2:.4f}")
    print(f"Macro F1: {f1_2:.4f}")
    print(f"Training time: {train_time_2:.2f}s")
    print(f"Inference time (test set): {inference_time_2:.4f}s")
    print(classification_report(y_test, y_pred_2_labels))

# ================================
# Comparison Summary
# ================================
print("\n" + "=" * 60)
print("TRADE-OFF ANALYSIS")
print("=" * 60)

if TRANSFORMERS_AVAILABLE:
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1", "Training Time (s)", "Inference Time (s)",
                    "Interpretability", "Model Size"],
        "TF-IDF + LinearSVC + LIME": [
            f"{acc_1:.3f}", f"{f1_1:.3f}", f"{train_time_1:.1f}",
            f"{inference_time_1:.4f}", "High (inherent + LIME)", "Small (<1MB)"
        ],
        "DistilBERT + Attention": [
            f"{acc_2:.3f}", f"{f1_2:.3f}", f"{train_time_2:.1f}",
            f"{inference_time_2:.4f}", "Medium (attention)", "Large (~260MB)"
        ]
    })
    print(comparison.to_string(index=False))
else:
    print(f"TF-IDF + LinearSVC Accuracy: {acc_1:.4f}")
    print("DistilBERT approach requires transformers/torch - install for full comparison.")
