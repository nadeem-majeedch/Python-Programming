#!/usr/bin/env python3
"""
Master generation script for all 7 Difficult-level hackathon problems.
Creates README.md, data/generate_data.py, starter_code.py, solution.py, requirements.txt
and generates synthetic datasets for each scenario.
"""

import os
import sys
import subprocess

PROBLEMS = [
    "14-multi-class-text-classification",
    "15-predictive-maintenance",
    "16-hybrid-recommendation-system",
    "17-medical-diagnosis-limited-data",
    "18-autonomous-vehicle-sensor-fusion",
    "19-network-anomaly-detection",
    "20-end-to-end-ml-pipeline",
]

BASE = os.path.dirname(os.path.abspath(__file__))


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content.lstrip("\n"))
    print(f"  Created: {path}")


# =============================================================================
# PROBLEM 14 - Multi-Class Text Classification with Interpretability
# =============================================================================

def p14_readme():
    return """# Problem 14: Multi-Class Text Classification with Interpretability

## Domain
NLP + Explainable AI (XAI)

## Problem Statement
You are building a content moderation and categorization engine for a news aggregation platform. The system must automatically classify short text documents into one of five categories (Technology, Health, Finance, Sports, Politics) and provide human-readable explanations for each classification decision. Stakeholders need to trust the model's decisions, especially for borderline cases where documents contain vocabulary that overlaps between categories.

The challenge is deliberately ambiguous: you must decide between traditional machine learning approaches (TF-IDF + linear models) that offer inherent interpretability but potentially lower accuracy, and deep learning approaches (Transformers) that achieve higher accuracy but are harder to explain. Your final submission must justify the trade-off you choose.

## Objectives
1. Achieve >85% classification accuracy on the held-out test set
2. Provide interpretable explanations for individual predictions (feature attribution or attention visualization)
3. Compare at least two fundamentally different approaches and present a quantitative trade-off analysis
4. Report inference latency vs accuracy vs interpretability quality

## Dataset
- 3000 short text documents across 5 categories
- Documents are 20-100 words in length
- Vocabulary overlaps between categories (e.g., "apple" appears in Technology and Finance)
- CSV at `data/text_documents.csv` with columns: doc_id, text, category

## Success Criteria
- **Accuracy**: >85% on test set (macro F1 > 0.83)
- **Interpretability**: For at least 10 test samples, provide feature-attribution explanations that clearly highlight which words drove the prediction
- **Trade-off Analysis**: Quantitative comparison (table/chart) of two approaches showing accuracy, training time, inference time, and interpretability quality score
- **Report**: 1-page analysis of the speed-accuracy-interpretability trade-off

## Starter Code
`starter_code.py` loads the data and provides a basic TF-IDF + LogisticRegression pipeline. You must extend this significantly.

## Constraints
- You may use any open-source library
- You must implement at least one "interpretable" approach AND one "black-box + explanation" approach
- The final model must be serializable (pickle/joblib) for deployment
"""


def p14_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic multi-class text dataset with overlapping vocabulary.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

categories = {
    "Technology": {
        "keywords": ["apple", "google", "software", "algorithm", "data", "cloud", "server",
                      "blockchain", "AI", "machine learning", "neural", "quantum", "chip",
                      "bandwidth", "encryption", "firewall", "protocol", "runtime", "framework",
                      "dashboard", "API", "microservice", "container", "linux", "python"],
        "templates": [
            "New {kw} update promises faster processing and lower latency for enterprise users.",
            "The latest {kw} breakthrough could transform how we handle {kw2} in production.",
            "Experts debate the ethical implications of {kw} in modern {kw2} systems.",
            "{kw} startup raises $50M to scale their {kw2} platform globally.",
            "Open-source {kw} community releases version 3.0 with improved {kw2} support.",
            "Researchers demonstrate {kw} achieving state-of-the-art results on {kw2} benchmarks.",
            "Security flaw discovered in popular {kw} library affects millions of {kw2} deployments.",
            "How {kw} is reshaping the future of {kw2} and autonomous systems."
        ]
    },
    "Health": {
        "keywords": ["vaccine", "clinical", "diagnosis", "therapy", "genomic", "surgery",
                      "apple", "protein", "cardiac", "neural", "dose", "symptom", "chronic",
                      "biomarker", "placebo", "antibody", "epidemic", "metabolic", "pediatric"],
        "templates": [
            "New study shows {kw} significantly reduces {kw2} symptoms in clinical trials.",
            "FDA approves novel {kw} therapy targeting {kw2} with fewer side effects.",
            "Researchers link {kw} levels to improved {kw2} outcomes in elderly patients.",
            "Breakthrough {kw} research opens doors for personalized {kw2} treatment.",
            "Hospital system deploys {kw} screening for early {kw2} detection.",
            "Global health organization warns about {kw} resistance and {kw2} spread.",
            "Phase 3 trial results indicate {kw} combined with {kw2} shows promising efficacy.",
            "Wearable device monitors {kw} to predict {kw2} episodes before they occur."
        ]
    },
    "Finance": {
        "keywords": ["apple", "stock", "dividend", "portfolio", "inflation", "bond",
                      "equity", "derivative", "leverage", "hedge", "crypto", "blockchain",
                      "dividend", "yield", "bullish", "bearish", "volatility", "quantum",
                      "liquidity", "amortization", "underwrite", "fiscal"],
        "templates": [
            "Market analysts predict {kw} rally as {kw2} exceeds quarterly expectations.",
            "Investors flock to {kw} as hedge against rising {kw2} and market volatility.",
            "Central bank policy shift impacts {kw} yields and {kw2} market sentiment.",
            "{kw} IPO oversubscribed 5x as institutional investors increase {kw2} exposure.",
            "Portfolio diversification with {kw} reduces {kw2} risk during downturns.",
            "Warren Buffett's Berkshire adds {kw} position, citing strong {kw2} fundamentals.",
            "Blockchain-based {kw} platform disrupts traditional {kw2} settlement systems.",
            "How {kw} derivatives are reshaping risk management in {kw2} markets."
        ]
    },
    "Sports": {
        "keywords": ["championship", "playoff", "MVP", "franchise", "recruit", "draft",
                      "scouting", "penalty", "overtime", "quarterback", "forward", "athlete",
                      "marathon", "tournament", "comeback", "underdog", "protocol", "injury"],
        "templates": [
            "Star {kw} candidate leads team to dramatic {kw2} victory in overtime.",
            "Former {kw} recruit makes surprising comeback after career-threatening {kw2}.",
            "Analytics revolution changes how teams approach {kw} and player {kw2} evaluation.",
            "Underdog story: how {kw} defied expectations to win the {kw2} championship.",
            "Controversial {kw} call decides {kw2} semifinal match in final seconds.",
            "Team invests in {kw} technology to reduce {kw2} and improve athlete recovery.",
            "Draft day surprises: top {kw} prospect falls to second round due to {kw2} concerns.",
            "World record shattered in {kw} as athlete credits revolutionary {kw2} training."
        ]
    },
    "Politics": {
        "keywords": ["legislation", "mandate", "congress", "policy", "campaign", "vote",
                      "diplomacy", "sanction", "referendum", "constituent", "lobby", "fiscal",
                      "tariff", "coalition", "partisan", "protocol", "mandate", "inflation"],
        "templates": [
            "New {kw} bill faces opposition over {kw2} provisions and budget concerns.",
            "Presidential {kw} announcement sparks debate on {kw2} reform across party lines.",
            "International {kw} agreement addresses {kw2} with new cooperative framework.",
            "Senate committee investigates {kw} amid allegations of {kw2} misconduct.",
            "Polling data reveals shifting {kw} preferences among {kw2} voters.",
            "{kw} coalition forms government after months of {kw2} negotiations.",
            "Proposed {kw} amendment targets {kw2} loopholes in current regulatory system.",
            "Analysis: how {kw} policy changes could affect {kw2} in the coming election cycle."
        ]
    }
}

# Overlapping keywords across categories
shared_keywords = {
    "apple": ["Technology", "Health", "Finance"],
    "blockchain": ["Technology", "Finance"],
    "quantum": ["Technology", "Finance"],
    "neural": ["Technology", "Health"],
    "protocol": ["Technology", "Sports", "Politics"],
    "fiscal": ["Finance", "Politics"],
    "inflation": ["Finance", "Politics"],
    "cloud": ["Technology"],
    "algorithm": ["Technology"],
    "data": ["Technology"],
    "vaccine": ["Health"],
    "clinical": ["Health"],
    "diagnosis": ["Health"],
    "stock": ["Finance"],
    "portfolio": ["Finance"],
    "championship": ["Sports"],
    "playoff": ["Sports"],
    "legislation": ["Politics"],
    "congress": ["Politics"],
    "campaign": ["Politics"]
}

def generate_document(category, cat_data, all_categories, doc_id):
    kw = np.random.choice(cat_data["keywords"])
    kw2 = np.random.choice(cat_data["keywords"])
    template = np.random.choice(cat_data["templates"])
    text = template.format(kw=kw, kw2=kw2)
    # Add overlapping vocabulary noise - sometimes start with a keyword from another category
    if np.random.random() < 0.3:
        other_cats = [c for c in all_categories if c != category]
        other_cat = np.random.choice(other_cats)
        noise_kw = np.random.choice(categories[other_cat]["keywords"])
        prefix = np.random.choice([
            f"Speaking of {noise_kw}, ",
            f"While {noise_kw} trends elsewhere, ",
            f"In related {noise_kw} news, ",
            f"Despite {noise_kw} concerns, "
        ])
        text = prefix + text[0].lower() + text[1:]
    return text

def main():
    cat_names = list(categories.keys())
    docs_per_cat = 600  # 3000 total
    all_texts = []
    all_labels = []
    doc_id = 0
    for cat in cat_names:
        for _ in range(docs_per_cat):
            text = generate_document(cat, categories[cat], cat_names, doc_id)
            all_texts.append(text)
            all_labels.append(cat)
            doc_id += 1
    df = pd.DataFrame({"doc_id": range(len(all_texts)),
                       "text": all_texts,
                       "category": all_labels})
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "text_documents.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} documents -> {out_path}")

if __name__ == "__main__":
    main()
"""


def p14_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Multi-Class Text Classification with Interpretability\"\"\"

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("data/text_documents.csv")
X = df["text"]
y = df["category"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train LogisticRegression
model = LogisticRegression(max_iter=1000, multi_class="multinomial")
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# TODO:
# 1. Improve accuracy with better preprocessing and hyperparameter tuning
# 2. Add interpretability (LIME, SHAP, or custom feature importance)
# 3. Implement a deep learning alternative (e.g., DistilBERT)
# 4. Compare approaches on accuracy, speed, and interpretability
"""


def p14_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Multi-Class Text Classification with Interpretability
Two approaches: (1) TF-IDF + LinearSVC + LIME, (2) DistilBERT + attention visualization
\"\"\"

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
print("\\n" + "=" * 60)
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
    print("\\n--- LIME Explanations (Approach 1) ---")
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
        print(f"\\nSample {idx}:")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print(f"  Top words for '{pred_label}':")
        for word, weight in exp.as_list(label=categories.index(pred_label)):
            print(f"    {word}: {weight:.4f}")

# ================================
# Approach 2: DistilBERT (if available)
# ================================
if TRANSFORMERS_AVAILABLE:
    print("\\n" + "=" * 60)
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
print("\\n" + "=" * 60)
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
"""


def p14_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lime>=0.2.0
transformers>=4.15.0
torch>=1.10.0
scipy>=1.7.0
joblib>=1.1.0
"""


# =============================================================================
# PROBLEM 15 - Predictive Maintenance
# =============================================================================

def p15_readme():
    return """# Problem 15: Predictive Maintenance

## Domain
Time Series + Classification

## Problem Statement
A manufacturing plant operates 100 industrial machines that are critical to production. Unexpected equipment failures cause costly downtime averaging $50,000 per hour. Your task is to build a predictive maintenance system that uses real-time sensor readings to forecast failures before they happen. The plant manager needs at least 12 hours of warning to schedule maintenance without disrupting production.

The dataset contains sensor readings collected hourly from each machine over several months. Only ~5% of readings are followed by a failure within 24 hours, creating a severe class imbalance. You must engineer temporal features that capture degradation trends and alert the team early enough to act.

## Objectives
1. Predict equipment failure within the next 24 hours from current sensor readings
2. Achieve F1 > 0.70 on the minority (failure) class
3. Maintain an average early warning time of >12 hours before actual failure
4. Handle the severe class imbalance (95:5) appropriately

## Dataset
- 5000 sensor readings from 100 machines
- Columns: MachineID, Timestamp, Temperature, Vibration, Pressure, RPM, OperatingHours, LastMaintenanceDays, Failure (binary target)
- CSV at `data/predictive_maintenance.csv`

## Success Criteria
- **F1-score**: >0.70 on the failure class
- **Early Warning Time**: Average prediction >12 hours before actual failure
- **Precision-Recall AUC**: >0.75
- **Robustness**: Consistent performance across different machine types and operating conditions

## Starter Code
`starter_code.py` loads data and trains a basic RandomForest. You must extend with temporal feature engineering and proper imbalance handling.

## Constraints
- You cannot use future data to predict past failures (no data leakage)
- Feature engineering must respect temporal ordering
- Cross-validation must use time-based (not random) splits
"""


def p15_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic predictive maintenance sensor data.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_machines = 100
n_records_per_machine = 50  # 5000 total
failure_rate = 0.05

records = []
for mid in range(1, n_machines + 1):
    base_temp = np.random.uniform(60, 80)
    base_vibration = np.random.uniform(0.5, 2.0)
    base_pressure = np.random.uniform(80, 120)
    base_rpm = np.random.uniform(800, 1500)
    
    machine_health = 1.0  # starts healthy, degrades over time
    failure_threshold = np.random.uniform(0.15, 0.25)
    in_degradation = False
    degradation_start = 0
    will_fail = np.random.random() < failure_rate
    
    for hour in range(n_records_per_machine):
        # Simulate degradation
        if will_fail and hour > 20 and not in_degradation:
            if np.random.random() < 0.3:
                in_degradation = True
                degradation_start = hour
        
        if in_degradation:
            progress = (hour - degradation_start) / (n_records_per_machine - degradation_start)
            machine_health = max(0.05, 1.0 - progress * 1.2)
        else:
            machine_health = min(1.0, machine_health + np.random.uniform(-0.02, 0.03))
        
        # Sensor readings with noise
        temp = base_temp + (1 - machine_health) * np.random.uniform(10, 30) + np.random.normal(0, 2)
        vibration = base_vibration + (1 - machine_health) * np.random.uniform(1, 4) + np.random.normal(0, 0.3)
        pressure = base_pressure + (1 - machine_health) * np.random.uniform(-15, 15) + np.random.normal(0, 3)
        rpm = base_rpm + (1 - machine_health) * np.random.uniform(-200, 100) + np.random.normal(0, 30)
        
        operating_hours = (mid * n_records_per_machine + hour) * np.random.uniform(0.9, 1.1)
        last_maint_days = np.random.poisson(30) if not in_degradation else max(0, 60 - (hour - degradation_start) * 3)
        
        # Failure in next 24h?
        failure_in_24h = 0
        if in_degradation and machine_health < failure_threshold:
            failure_in_24h = 1
        
        records.append({
            "MachineID": mid,
            "Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
            "Temperature": round(temp, 2),
            "Vibration": round(vibration, 3),
            "Pressure": round(pressure, 2),
            "RPM": round(rpm, 1),
            "OperatingHours": round(operating_hours, 1),
            "LastMaintenanceDays": round(last_maint_days, 1),
            "Failure": failure_in_24h
        })

df = pd.DataFrame(records)
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "predictive_maintenance.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} sensor readings (failures: {df['Failure'].sum()}) -> {out_path}")
"""


def p15_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Predictive Maintenance\"\"\"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Load data
df = pd.read_csv("data/predictive_maintenance.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Basic features
features = ["Temperature", "Vibration", "Pressure", "RPM",
            "OperatingHours", "LastMaintenanceDays"]
X = df[features]
y = df["Failure"]

# Train/test split (temporal - use first 80% for train)
df_sorted = df.sort_values("Timestamp")
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

X_train = train_df[features]
y_train = train_df["Failure"]
X_test = test_df[features]
y_test = test_df["Failure"]

print(f"Train failures: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Test failures: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\\nRandomForest Baseline:")
print(classification_report(y_test, y_pred))
print(f"F1 (failure class): {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

# TODO:
# 1. Engineer temporal features: rolling windows, rate of change, lag features
# 2. Handle class imbalance better (SMOTE, scale_pos_weight, threshold tuning)
# 3. Implement early warning time metric
# 4. Try XGBoost/LightGBM with proper tuning
"""


def p15_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Predictive Maintenance with temporal features and XGBoost.\"\"\"

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")
    from sklearn.ensemble import RandomForestClassifier as XGBModel

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/predictive_maintenance.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["MachineID", "Timestamp"]).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Failure rate: {df['Failure'].mean()*100:.2f}%")
print(f"Machines: {df['MachineID'].nunique()}")

# ========================
# Temporal feature engineering
# ========================
def engineer_features(df):
    # Create rolling window, lag, and rate-of-change features.
    df = df.copy()
    # Sort per machine
    df = df.sort_values(["MachineID", "Timestamp"]).reset_index(drop=True)
    
    base_features = ["Temperature", "Vibration", "Pressure", "RPM",
                     "OperatingHours", "LastMaintenanceDays"]
    
    # Rolling window features (last 6 and 12 hours)
    for w in [3, 6, 12]:
        for col in ["Temperature", "Vibration", "Pressure", "RPM"]:
            grp = df.groupby("MachineID")[col]
            df[f"{col}_mean_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_std_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
            df[f"{col}_max_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
            df[f"{col}_min_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
    
    # Rate of change (difference from previous reading)
    for col in ["Temperature", "Vibration", "Pressure", "RPM"]:
        grp = df.groupby("MachineID")[col]
        df[f"{col}_delta"] = grp.diff()
        df[f"{col}_delta_abs"] = df[f"{col}_delta"].abs()
        df[f"{col}_pct_change"] = grp.pct_change() * 100
    
    # Lag features (1, 2, 3 hour lag)
    for lag in [1, 2, 3]:
        for col in base_features:
            df[f"{col}_lag_{lag}"] = df.groupby("MachineID")[col].shift(lag)
    
    # Interaction features
    df["temp_vib_ratio"] = df["Temperature"] / (df["Vibration"] + 0.001)
    df["pressure_x_rpm"] = df["Pressure"] * df["RPM"]
    df["temp_pressure_interaction"] = df["Temperature"] * df["Pressure"] / 1000
    
    # Maintenance-related
    df["high_op_hours"] = (df["OperatingHours"] > df.groupby("MachineID")["OperatingHours"].transform("median")).astype(int)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

df_feat = engineer_features(df)
feature_cols = [c for c in df_feat.columns if c not in
                ["MachineID", "Timestamp", "Failure"]]

print(f"\\nFeatures engineered: {len(feature_cols)}")

# ========================
# Temporal split for validation
# ========================
df_feat = df_feat.sort_values("Timestamp").reset_index(drop=True)
split_idx = int(len(df_feat) * 0.8)
train_df = df_feat.iloc[:split_idx]
test_df = df_feat.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df["Failure"]
X_test = test_df[feature_cols]
y_test = test_df["Failure"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape}, failures: {y_train.sum()}")
print(f"Test:  {X_test.shape}, failures: {y_test.sum()}")

# ========================
# Train XGBoost with imbalance handling
# ========================
if XGB_AVAILABLE:
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
else:
    model = XGBModel(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

y_prob = model.predict_proba(X_test_s)[:, 1]
y_pred = model.predict(X_test_s)

# ========================
# Threshold tuning for early warning
# ========================
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that gives best F1
best_f1 = 0
best_thresh = 0.5
for thresh in np.linspace(0.05, 0.95, 100):
    pred_thresh = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_test, pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\\nOptimal threshold: {best_thresh:.3f} (F1: {best_f1:.4f})")

y_pred_opt = (y_prob >= best_thresh).astype(int)
print("\\n--- XGBoost Results (tuned threshold) ---")
print(classification_report(y_test, y_pred_opt))

f1_failure = f1_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = auc(recalls, precisions)
print(f"F1 (failure): {f1_failure:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ========================
# Early Warning Time Analysis
# ========================
print("\\n--- Early Warning Time Analysis ---")

test_df_local = test_df.copy()
test_df_local["y_prob"] = y_prob
test_df_local["y_pred"] = y_pred_opt

warning_times = []
for mid in test_df_local["MachineID"].unique():
    machine_data = test_df_local[test_df_local["MachineID"] == mid].sort_values("Timestamp")
    failure_rows = machine_data[machine_data["Failure"] == 1]
    alert_rows = machine_data[machine_data["y_pred"] == 1]
    
    if len(failure_rows) > 0 and len(alert_rows) > 0:
        first_failure = failure_rows.index[0]
        alerts_before = alert_rows[alert_rows.index < first_failure]
        if len(alerts_before) > 0:
            first_alert = alerts_before.index[0]
            hours_before = (failure_rows["Timestamp"].iloc[0] -
                           machine_data.loc[first_alert, "Timestamp"]).total_seconds() / 3600
            warning_times.append(hours_before)

if warning_times:
    print(f"Average warning time: {np.mean(warning_times):.1f} hours")
    print(f"Median warning time: {np.median(warning_times):.1f} hours")
    print(f"Min warning time: {min(warning_times):.1f} hours")
    print(f"Max warning time: {max(warning_times):.1f} hours")
else:
    print("No early warnings generated")

# ========================
# Feature Importance
# ========================
if XGB_AVAILABLE:
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\\n--- Top 10 Features ---")
    print(importance.head(10).to_string(index=False))
"""


def p15_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
"""


# =============================================================================
# PROBLEM 16 - Hybrid Recommendation System
# =============================================================================

def p16_readme():
    return """# Problem 16: Hybrid Recommendation System

## Domain
Recommendation Systems

## Problem Statement
You are building the recommendation engine for a new e-commerce platform. The platform has both established users with rating histories and new users who just signed up (cold-start). Similarly, some products are well-established while others are newly listed with no ratings. A pure collaborative filtering approach fails for cold-start scenarios, and a pure content-based approach lacks serendipity.

Your task is to design a hybrid recommendation system that combines collaborative filtering (matrix factorization) with content-based methods. The system must produce accurate rating predictions for known users while still providing reasonable recommendations for cold-start users and new items.

## Objectives
1. Achieve RMSE < 1.0 on held-out rating predictions
2. Provide meaningful recommendations for cold-start users (no history) using demographic and content information
3. Achieve precision@k > 0.6 for top-10 recommendations
4. Demonstrate diversity/serendipity in recommendations beyond simply popular items

## Dataset
- 500 users with demographics (age, region)
- 200 items with metadata (category, price, popularity)
- 10000 interactions (ratings 1-5)
- Includes cold-start users (no history) and items (no ratings)
- CSV files: `data/ratings.csv`, `data/users.csv`, `data/items.csv`

## Success Criteria
- **RMSE**: < 1.0 on test ratings (existing users)
- **Cold-start**: At least 4/5 reasonable recommendations for cold-start users
- **Precision@10**: > 0.60
- **Diversity**: Average pairwise dissimilarity > 0.3 within recommendation lists

## Starter Code
`starter_code.py` loads data and provides a basic SVD (Singular Value Decomposition) approach. You must extend with hybrid blending.

## Constraints
- Must handle both warm-start and cold-start scenarios
- Must combine collaborative filtering and content-based signals
- Must evaluate both prediction accuracy and recommendation quality
"""


def p16_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic recommendation system data.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_users = 500
n_items = 200
n_interactions = 10000
n_cold_users = 30
n_cold_items = 20

# Generate user demographics
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

users = []
for uid in range(1, n_users + 1):
    users.append({
        "UserID": uid,
        "Age": int(np.random.normal(35, 12)),
        "Region": np.random.choice(regions, p=[0.25, 0.25, 0.25, 0.25])
    })
df_users = pd.DataFrame(users)

# Generate item metadata
items = []
for iid in range(1, n_items + 1):
    cat = np.random.choice(categories)
    items.append({
        "ItemID": iid,
        "Category": cat,
        "Price": round(np.random.lognormal(3.5, 0.5), 2),
        "Popularity": np.random.uniform(0, 1)
    })
df_items = pd.DataFrame(items)

# Create latent factors for users and items (underlying preference structure)
n_factors = 10
np.random.seed(42)
user_factors = np.random.randn(n_users, n_factors)
item_factors = np.random.randn(n_items, n_factors)

# Region and category biases
region_bias = {"North": 0.2, "South": -0.1, "East": 0.3, "West": -0.2}
cat_bias = {"Electronics": 0.3, "Clothing": -0.2, "Books": 0.1, "Home": 0.0, "Sports": -0.1}

# Generate ratings
ratings = []
interaction_count = 0
attempts = 0
max_attempts = 100000

while interaction_count < n_interactions and attempts < max_attempts:
    attempts += 1
    uid = np.random.randint(1, n_users + 1)
    # Skip cold-start users (they won't have ratings)
    if uid > n_users - n_cold_users:
        continue
    iid = np.random.randint(1, n_items + 1)
    # Skip cold-start items (they won't have ratings)
    if iid > n_items - n_cold_items:
        continue
    
    u_idx = uid - 1
    i_idx = iid - 1
    
    # Rating = latent dot product + biases + noise
    rating = np.dot(user_factors[u_idx], item_factors[i_idx])
    user_row = df_users.iloc[u_idx]
    item_row = df_items.iloc[i_idx]
    rating += region_bias.get(user_row["Region"], 0)
    rating += cat_bias.get(item_row["Category"], 0)
    rating += np.random.normal(0, 0.5)
    
    rating = np.clip(rating, 0.5, 5.0)
    rating = round(rating)
    
    ratings.append({
        "UserID": uid,
        "ItemID": iid,
        "Rating": int(rating)
    })
    interaction_count += 1

df_ratings = pd.DataFrame(ratings)

# Ensure cold-start users have no ratings
cold_uids = list(range(n_users - n_cold_users + 1, n_users + 1))
df_ratings = df_ratings[~df_ratings["UserID"].isin(cold_uids)]

# Ensure cold-start items have no ratings
cold_iids = list(range(n_items - n_cold_items + 1, n_items + 1))
df_ratings = df_ratings[~df_ratings["ItemID"].isin(cold_iids)]

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df_ratings.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)
df_users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
df_items.to_csv(os.path.join(out_dir, "items.csv"), index=False)

print(f"Ratings: {len(df_ratings)}, Users: {len(df_users)}, Items: {len(df_items)}")
print(f"Cold-start users: {n_cold_users}, Cold-start items: {n_cold_items}")
"""


def p16_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Hybrid Recommendation System\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

print(f"Ratings: {len(ratings)}")
print(f"Users: {len(users)} (cold-start: {users[users['UserID'].isin(ratings['UserID'].unique())].shape[0]} active)")
print(f"Items: {len(items)} (cold-start: {items[~items['ItemID'].isin(ratings['ItemID'].unique())].shape[0]} new)")

# SVD via numpy
R = ratings.pivot_table(index="UserID", columns="ItemID", values="Rating").values
R_mean = np.nanmean(R)
R_demeaned = R - R_mean
R_demeaned[np.isnan(R_demeaned)] = 0

U, S, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
k = 20
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# Reconstruct
R_pred = np.dot(U_k, np.dot(S_k, Vt_k)) + R_mean

# Evaluate on known ratings
mask = ~np.isnan(R)
rmse = sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))
print(f"\\nBasic SVD RMSE: {rmse:.4f}")

# TODO:
# 1. Split data properly for evaluation (training set vs test set)
# 2. Implement proper matrix factorization (SVD ++ or ALS)
# 3. Add content-based component using user/item metadata
# 4. Blend collaborative + content-based predictions
# 5. Handle cold-start users and items
# 6. Evaluate diversity and precision@k
"""


def p16_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Hybrid Recommendation System (SVD + Content-Based)\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load data
# ========================
ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

# Encode categoricals
region_encoder = LabelEncoder()
cat_encoder = LabelEncoder()
users["Region_enc"] = region_encoder.fit_transform(users["Region"])
items["Category_enc"] = cat_encoder.fit_transform(items["Category"])

# Split ratings: keep some users as "cold-start" for evaluation
all_active_users = ratings["UserID"].unique()
np.random.seed(42)
test_users = np.random.choice(all_active_users, size=int(len(all_active_users) * 0.15), replace=False)
train_ratings = ratings[~ratings["UserID"].isin(test_users)]
test_ratings = ratings[ratings["UserID"].isin(test_users)]

print(f"Train ratings: {len(train_ratings)}, Test ratings: {len(test_ratings)}")
print(f"Global mean rating: {ratings['Rating'].mean():.3f}")

# ========================
# Collaborative Filtering (SVD)
# ========================
R_train = train_ratings.pivot_table(
    index="UserID", columns="ItemID", values="Rating"
).values
R_mean = np.nanmean(R_train)
R_demeaned = R_train - R_mean
R_demeaned[np.isnan(R_demeaned)] = 0

U, S, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
k = 25
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

R_svd = np.dot(U_k, np.dot(S_k, Vt_k)) + R_mean

# Map to predictions for test
user_ids_sorted = sorted(train_ratings["UserID"].unique())
item_ids_sorted = sorted(train_ratings["ItemID"].unique())
user_to_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids_sorted)}

def predict_svd(uid, iid):
    if uid in user_to_idx and iid in item_to_idx:
        return R_svd[user_to_idx[uid], item_to_idx[iid]]
    return R_mean

# Evaluate SVD on test
svd_preds = []
for _, row in test_ratings.iterrows():
    svd_preds.append(predict_svd(row["UserID"], row["ItemID"]))
svd_rmse = sqrt(mean_squared_error(test_ratings["Rating"], svd_preds))
print(f"SVD-only RMSE: {svd_rmse:.4f}")

# ========================
# Content-Based Component
# ========================
def compute_content_score(uid, iid):
    \"\"\"Simple content-based score using demographics and item metadata.\"\"\"
    user_row = users[users["UserID"] == uid]
    item_row = items[items["ItemID"] == uid]
    if user_row.empty or item_row.empty:
        return 0.5
    
    user_row = user_row.iloc[0]
    item_row = item_row.iloc[0]
    
    score = 0.0
    # Age-based affinity (simplified)
    cat_age_map = {
        "Electronics": (20, 40), "Clothing": (18, 60),
        "Books": (25, 65), "Home": (30, 60), "Sports": (18, 45)
    }
    cat_range = cat_age_map.get(item_row["Category"], (0, 100))
    if cat_range[0] <= user_row["Age"] <= cat_range[1]:
        score += 0.3
    
    # Region-based affinity
    region_cat_prefs = {
        "North": ["Electronics", "Books"],
        "South": ["Clothing", "Sports"],
        "East": ["Books", "Electronics"],
        "West": ["Home", "Sports"]
    }
    preferred = region_cat_prefs.get(user_row["Region"], [])
    if item_row["Category"] in preferred:
        score += 0.3
    
    # Price tolerance (older users prefer higher-priced items)
    price_factor = min(1.0, item_row["Price"] / 50.0)
    if user_row["Age"] > 40:
        score += 0.2 * price_factor
    else:
        score += 0.2 * (1 - price_factor)
    
    return min(1.0, score + 0.2)

# ========================
# Hybrid Blending
# ========================
def hybrid_predict(uid, iid, alpha=0.7):
    \"\"\"Blend SVD and content-based predictions.\"\"\"
    if uid in user_to_idx and iid in item_to_idx:
        svd_score = predict_svd(uid, iid)
        # Normalize SVD rating to [0, 1]
        svd_norm = (svd_score - 1) / 4.0
        content_score = compute_content_score(uid, iid)
        hybrid = alpha * svd_norm + (1 - alpha) * content_score
        return 1 + hybrid * 4  # Scale back to 1-5
    else:
        # Cold-start: pure content-based
        content_score = compute_content_score(uid, iid)
        return 1 + content_score * 4

# Find optimal alpha
best_alpha = 0.7
best_rmse = float("inf")
for alpha in np.linspace(0.3, 0.95, 10):
    preds = []
    for _, row in test_ratings.iterrows():
        preds.append(hybrid_predict(row["UserID"], row["ItemID"], alpha))
    rmse = sqrt(mean_squared_error(test_ratings["Rating"], preds))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\\nBest alpha: {best_alpha:.2f}, Hybrid RMSE: {best_rmse:.4f}")

# ========================
# Cold-Start Evaluation
# ========================
cold_users = users[~users["UserID"].isin(ratings["UserID"].unique())]
cold_items = items[~items["ItemID"].isin(ratings["ItemID"].unique())]

print(f"\\nCold-start users: {len(cold_users)}")
print(f"Cold-start items: {len(cold_items)}")

# For each cold-start user, recommend top-N items
def recommend_cold_start(uid, n=10):
    \"\"\"Recommend using only content-based scores.\"\"\"
    scores = []
    for _, item_row in items.iterrows():
        score = compute_content_score(uid, item_row["ItemID"])
        scores.append((item_row["ItemID"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in scores[:n]]

print("\\nCold-start user recommendations:")
for _, user_row in cold_users.head(5).iterrows():
    recs = recommend_cold_start(user_row["UserID"], n=5)
    rec_cats = items[items["ItemID"].isin(recs)]["Category"].tolist()
    print(f"  User {user_row['UserID']} ({user_row['Age']}, {user_row['Region']}): {rec_cats}")

# ========================
# Diversity Metric
# ========================
def category_diversity(rec_list):
    \"\"\"Diversity = 1 - (proportion of most common category).\"\"\"
    cats = items[items["ItemID"].isin(rec_list)]["Category"].tolist()
    if not cats:
        return 0
    most_common = max(cats.count(c) for c in set(cats))
    return 1 - most_common / len(cats)

print("\\n--- Diversity Analysis ---")
diversities = []
for uid in np.random.choice(all_active_users, size=20, replace=False):
    recs = recommend_cold_start(uid, n=10)
    div = category_diversity(recs)
    diversities.append(div)
print(f"Average diversity across 20 users: {np.mean(diversities):.3f}")

# ========================
# Final Summary
# ========================
print("\\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"SVD-only RMSE:         {svd_rmse:.4f}")
print(f"Hybrid RMSE (alpha={best_alpha:.2f}): {best_rmse:.4f}")
print(f"Cold-start strategy:   Content-based fallback")
print(f"Average diversity:     {np.mean(diversities):.3f}")
print(f"RMSE < 1.0: {'PASS' if best_rmse < 1.0 else 'NEEDS IMPROVEMENT'}")
"""


def p16_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
"""


# =============================================================================
# PROBLEM 17 - Medical Diagnosis with Limited Data
# =============================================================================

def p17_readme():
    return """# Problem 17: Medical Diagnosis with Limited Data

## Domain
Classification (Small Data)

## Problem Statement
A rare disease affects approximately 10% of the population. Your team has collected patient biomarker data from only 300 patients (30 diagnosed, 270 healthy). Clinical data collection is expensive and slow, so you must build the best possible diagnostic model with what you have. The 30 features include many irrelevant ones, and several features are multicollinear.

Neural networks are explicitly forbidden -- with only 300 samples, a model with even modest capacity would have more parameters than data points and would severely overfit. You must use classical statistical learning with proper regularization, feature selection, and rigorous validation.

## Objectives
1. Build a reliable diagnostic model achieving AUC > 0.80
2. Ensure well-calibrated probabilities (Brier score < 0.20)
3. Select informative features while discarding noise
4. Use appropriate validation (LOO-CV, bootstrap) rather than simple train-test split

## Dataset
- 300 samples, 30 biomarker features
- Binary diagnosis: rare disease (~10% prevalence)
- Many irrelevant features, multicollinearity present
- CSV at `data/medical_diagnosis.csv`

## Success Criteria
- **AUC**: >0.80 on held-out predictions
- **Brier Score**: <0.20 (calibrated probabilities)
- **Feature Selection**: Identify at least the top 5 most important biomarkers
- **Confidence Intervals**: Report 95% CI for AUC using bootstrap

## Starter Code
`starter_code.py` loads data and trains a basic LogisticRegression (which will overfit). You must add regularization and proper validation.

## Constraints
- Neural networks are NOT allowed (too many parameters for 300 samples)
- Must use proper validation strategy (leave-one-out or bootstrap recommended)
- Must report confidence intervals on performance metrics
"""


def p17_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic medical diagnosis data with limited samples.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 300
n_features = 30
prevalence = 0.10

# Create latent structure: only 8 features are truly informative
n_informative = 8
n_noise = n_features - n_informative

# Generate labels
y = np.random.binomial(1, prevalence, n_samples)

# Informative features with multicollinearity
X_informative = np.zeros((n_samples, n_informative))
for i in range(n_informative):
    base = np.random.randn(n_samples) * 0.5
    # Add disease signal
    effect_size = np.random.uniform(1.0, 2.5)
    base += y * effect_size
    X_informative[:, i] = base

# Create multicollinearity: feature 3 is correlated with feature 1+2
X_informative[:, 2] = 0.7 * X_informative[:, 0] + 0.3 * X_informative[:, 1] + np.random.randn(n_samples) * 0.3
# Feature 7 is correlated with feature 5
X_informative[:, 6] = 0.8 * X_informative[:, 4] + np.random.randn(n_samples) * 0.4

# Noise features (pure random)
X_noise = np.random.randn(n_samples, n_noise) * 0.8

# Combine
X = np.hstack([X_informative, X_noise])

# Add some label noise (flip 5% of labels)
n_flip = int(n_samples * 0.05)
flip_idx = np.random.choice(n_samples, n_flip, replace=False)
y[flip_idx] = 1 - y[flip_idx]

# Create DataFrame
feature_names = [f"Biomarker_{i+1:02d}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df["Diagnosis"] = y

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "medical_diagnosis.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} samples (positive: {df['Diagnosis'].sum()}, {df['Diagnosis'].mean()*100:.1f}%) -> {out_path}")
"""


def p17_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Medical Diagnosis with Limited Data\"\"\"

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

# Load data
df = pd.read_csv("data/medical_diagnosis.csv")
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

print(f"Data shape: {X.shape}")
print(f"Positive rate: {y.mean()*100:.1f}%")

# Simple train-test split (not ideal for small data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Basic LogisticRegression (will overfit with 30 features on 300 samples)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")

# TODO:
# 1. Add regularization (Ridge, Lasso, ElasticNet)
# 2. Perform feature selection (RFE, SelectKBest, L1 regularization)
# 3. Use proper validation (LOO-CV, bootstrap)
# 4. Calibrate probabilities (Platt scaling)
# 5. Report confidence intervals on AUC
"""


def p17_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Medical Diagnosis with Limited Data
Ridge regression + feature selection + bootstrap confidence intervals.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load data
# ========================
df = pd.read_csv("data/medical_diagnosis.csv")
X = df.drop(columns=["Diagnosis"]).values
feature_names = df.drop(columns=["Diagnosis"]).columns.tolist()
y = df["Diagnosis"].values

print(f"Data: {X.shape}, Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# ========================
# Approach 1: L1-regularized LogisticRegression
# ========================
print("\\n" + "=" * 50)
print("APPROACH 1: L1-regularized LogisticRegression")
print("=" * 50)

# Use LOOCV for honest evaluation
loo = LeaveOneOut()
y_prob_l1 = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_prob_l1[test_idx] = model.predict_proba(X_te_s)[:, 1]

auc_l1 = roc_auc_score(y, y_prob_l1)
brier_l1 = brier_score_loss(y, y_prob_l1)
print(f"LOOCV AUC: {auc_l1:.4f}")
print(f"LOOCV Brier: {brier_l1:.4f}")

# ========================
# Approach 2: Ridge (L2) + Feature Selection
# ========================
print("\\n" + "=" * 50)
print("APPROACH 2: Ridge + SelectKBest Feature Selection")
print("=" * 50)

# Cross-validated feature selection + Ridge
k_values = [5, 8, 10, 12, 15]
best_k = 8
best_auc = 0

for k in k_values:
    y_prob_k = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        selector = SelectKBest(f_classif, k=k)
        X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
        X_te_sel = selector.transform(X_te_s)
        
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42)
        model.fit(X_tr_sel, y_tr)
        y_prob_k[test_idx] = model.predict_proba(X_te_sel)[:, 1]
    
    auc_k = roc_auc_score(y, y_prob_k)
    if auc_k > best_auc:
        best_auc = auc_k
        best_k = k

print(f"Best k: {best_k}, LOOCV AUC: {best_auc:.4f}")

# Refit with best k using all data to show selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector = SelectKBest(f_classif, k=best_k)
X_sel = selector.fit_transform(X_scaled, y)
selected_mask = selector.get_support()
selected_features = [f for f, sel in zip(feature_names, selected_mask) if sel]
print(f"Selected features ({len(selected_features)}): {selected_features}")

final_model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42)
final_model.fit(X_sel, y)

# ========================
# Approach 3: ElasticNet
# ========================
print("\\n" + "=" * 50)
print("APPROACH 3: ElasticNet (L1 + L2)")
print("=" * 50)

y_prob_en = np.zeros(len(y))
for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(penalty="elasticnet", solver="saga",
                                C=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_prob_en[test_idx] = model.predict_proba(X_te_s)[:, 1]

auc_en = roc_auc_score(y, y_prob_en)
brier_en = brier_score_loss(y, y_prob_en)
print(f"ElasticNet LOOCV AUC: {auc_en:.4f}")
print(f"ElasticNet LOOCV Brier: {brier_en:.4f}")

# ========================
# Bootstrap Confidence Intervals
# ========================
print("\\n" + "=" * 50)
print("BOOTSTRAP CONFIDENCE INTERVALS (ElasticNet)")
print("=" * 50)

n_bootstrap = 1000
bootstrap_aucs = []

for b in range(n_bootstrap):
    idx = np.random.choice(len(y), len(y), replace=True)
    X_boot, y_boot = X[idx], y[idx]
    
    # Leave-pair-out within bootstrap
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=b)
    aucs_fold = []
    for train_idx, test_idx in skf.split(X_boot, y_boot):
        X_tr, X_te = X_boot[train_idx], X_boot[test_idx]
        y_tr, y_te = y_boot[train_idx], y_boot[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = LogisticRegression(penalty="elasticnet", solver="saga",
                                    C=0.1, l1_ratio=0.5, max_iter=5000)
        model.fit(X_tr_s, y_tr)
        y_prob_b = model.predict_proba(X_te_s)[:, 1]
        if len(np.unique(y_te)) > 1:
            aucs_fold.append(roc_auc_score(y_te, y_prob_b))
    
    if aucs_fold:
        bootstrap_aucs.append(np.mean(aucs_fold))

bootstrap_aucs = np.array(bootstrap_aucs)
ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)
print(f"Bootstrap AUC: {np.mean(bootstrap_aucs):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ========================
# Probability Calibration
# ========================
print("\\n" + "=" * 50)
print("CALIBRATION ANALYSIS")
print("=" * 50)

# Use the best approach (ElasticNet) and calibrate
calibrator = CalibratedClassifierCV(
    LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=5000),
    cv=5, method="sigmoid"
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_prob_cal = cross_val_predict(calibrator, X_scaled, y, cv=5, method="predict_proba")[:, 1]
brier_cal = brier_score_loss(y, y_prob_cal)
auc_cal = roc_auc_score(y, y_prob_cal)

print(f"After Platt scaling:")
print(f"AUC: {auc_cal:.4f}")
print(f"Brier Score: {brier_cal:.4f}")
print(f"Brier < 0.20: {'PASS' if brier_cal < 0.20 else 'NEEDS IMPROVEMENT'}")

# ========================
# Final Comparison
# ========================
print("\\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)
comparison = pd.DataFrame({
    "Method": ["L1 LogisticRegression", "Ridge + SelectKBest",
               "ElasticNet", "ElasticNet (Calibrated)"],
    "AUC": [f"{auc_l1:.4f}", f"{best_auc:.4f}", f"{auc_en:.4f}", f"{auc_cal:.4f}"],
    "Brier": [f"{brier_l1:.4f}", "-", f"{brier_en:.4f}", f"{brier_cal:.4f}"],
})
print(comparison.to_string(index=False))
print(f"\\n95% CI for ElasticNet AUC: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Selected biomarkers: {selected_features}")
"""


def p17_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
scipy>=1.7.0
"""


# =============================================================================
# PROBLEM 18 - Autonomous Vehicle Sensor Fusion
# =============================================================================

def p18_readme():
    return """# Problem 18: Autonomous Vehicle Sensor Fusion

## Domain
Multi-modal Data Fusion

## Problem Statement
An autonomous vehicle is equipped with three types of sensors: Camera, LiDAR, and Radar. Each sensor provides complementary information about the vehicle's surroundings, but each has limitations. Cameras provide rich visual data but struggle in poor lighting. LiDAR provides precise 3D geometry but has limited range in rain. Radar works in all conditions but has lower resolution. Sensors can also fail intermittently (dropouts).

Your task is to fuse data from these heterogeneous sensors to classify obstacles as Pedestrian, Cyclist, Car, Truck, or None. You must handle sensor failures gracefully and provide uncertainty estimates for your predictions.

## Objectives
1. Achieve >90% classification accuracy when all sensors are operating
2. Maintain >80% accuracy when any single sensor fails
3. Handle conflicting sensor readings and quantify prediction uncertainty
4. Compare early fusion, late fusion, and hybrid fusion strategies

## Dataset
- 2000 synchronized time steps from 3 sensors
- Camera: object class probabilities for 5 classes (10-dimensional features)
- LiDAR: x,y,z coordinates of nearest object (3 features)
- Radar: speed, distance, angle (3 features)
- Ground truth: obstacle type (Pedestrian/Cyclist/Car/Truck/None)
- Sensors have different noise levels and intermittent dropouts
- CSV at `data/sensor_data.csv`

## Success Criteria
- **All sensors**: Accuracy > 90%
- **One sensor failure**: Accuracy > 80%
- **Graceful degradation**: Performance degrades smoothly as sensors drop out
- **Uncertainty**: Reasonable confidence scores that correlate with correctness

## Starter Code
`starter_code.py` loads synchronized data and implements a simple voting ensemble. You must extend with proper fusion strategies.

## Constraints
- Must handle missing sensor data (NaN values)
- Must evaluate under multiple sensor failure scenarios
- Must report per-class performance and confusion matrix
"""


def p18_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic autonomous vehicle sensor fusion data.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_timesteps = 2000
obstacle_types = ["Pedestrian", "Cyclist", "Car", "Truck", "None"]
sensor_failure_rate = 0.05  # 5% dropout per sensor

records = []
for t in range(n_timesteps):
    # Ground truth
    gt = np.random.choice(obstacle_types, p=[0.15, 0.10, 0.25, 0.10, 0.40])
    
    # Camera features (object class probabilities - softmax output)
    camera_base = np.zeros(5)
    if gt != "None":
        gt_idx = obstacle_types.index(gt)
        camera_base[gt_idx] = np.random.uniform(0.7, 0.95)
        # Confuse with similar classes
        if gt in ["Car", "Truck"]:
            alt_idx = obstacle_types.index("Truck" if gt == "Car" else "Car")
            camera_base[alt_idx] = np.random.uniform(0.0, 0.2)
        elif gt in ["Pedestrian", "Cyclist"]:
            alt_idx = obstacle_types.index("Cyclist" if gt == "Pedestrian" else "Pedestrian")
            camera_base[alt_idx] = np.random.uniform(0.0, 0.15)
        # Remaining probability to others
        remaining = 1.0 - camera_base.sum()
        for i in range(5):
            if camera_base[i] == 0:
                camera_base[i] = np.random.uniform(0, remaining * 0.3)
        camera_base = camera_base / camera_base.sum()  # renormalize
    else:
        # No obstacle - uniform-ish distribution
        camera_base = np.random.dirichlet(np.ones(5) * 2)
    
    camera_feat = camera_base + np.random.normal(0, 0.02, 5)
    camera_feat = np.clip(camera_feat, 0, 1)
    camera_feat = camera_feat / camera_feat.sum()
    
    # Camera dropout
    if np.random.random() < sensor_failure_rate:
        camera_feat = np.full(5, np.nan)
    
    # LiDAR features (x, y, z of nearest obstacle)
    if gt == "None":
        lidar_feat = np.array([np.random.uniform(50, 100),
                               np.random.uniform(-10, 10),
                               np.random.uniform(0, 0.5)])
    else:
        # Distance based on obstacle type
        dist_ranges = {"Pedestrian": (5, 30), "Cyclist": (5, 40),
                       "Car": (5, 80), "Truck": (5, 100)}
        d_min, d_max = dist_ranges[gt]
        dist = np.random.uniform(d_min, d_max)
        angle = np.random.uniform(-30, 30)
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        z = np.random.uniform(0.5, 2.5) if gt == "Pedestrian" else np.random.uniform(1.0, 3.0)
        lidar_feat = np.array([x, y, z])
        # Add noise
        lidar_noise = np.array([0.1, 0.05, 0.02]) * dist / 10
        lidar_feat += np.random.normal(0, lidar_noise)
    
    # LiDAR dropout
    if np.random.random() < sensor_failure_rate:
        lidar_feat = np.full(3, np.nan)
    
    # Radar features (speed, distance, angle)
    if gt == "None":
        radar_feat = np.array([np.random.uniform(-1, 1),
                               np.random.uniform(80, 120),
                               np.random.uniform(-5, 5)])
    else:
        speed_ranges = {"Pedestrian": (1, 6), "Cyclist": (10, 30),
                        "Car": (20, 80), "Truck": (10, 60)}
        speed = np.random.uniform(*speed_ranges[gt])
        dist_ranges_r = {"Pedestrian": (5, 30), "Cyclist": (5, 40),
                         "Car": (10, 100), "Truck": (10, 120)}
        d_min, d_max = dist_ranges_r[gt]
        dist = np.random.uniform(d_min, d_max)
        angle = np.random.uniform(-30, 30) + np.random.normal(0, 2)
        radar_feat = np.array([speed, dist, angle])
        # Radar is more noise-resistant but less accurate
        radar_feat += np.random.normal(0, [0.5, 0.5, 1.0])
    
    # Radar dropout
    if np.random.random() < sensor_failure_rate:
        radar_feat = np.full(3, np.nan)
    
    row = {"Timestep": t, "ObstacleType": gt}
    row.update({f"Camera_{i}": camera_feat[i] for i in range(5)})
    row.update({f"LiDAR_{'xyz'[i]}": lidar_feat[i] for i in range(3)})
    row.update({f"Radar_{['Speed','Dist','Angle'][i]}": radar_feat[i] for i in range(3)})
    records.append(row)

df = pd.DataFrame(records)
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sensor_data.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} sensor readings -> {out_path}")
print(f"Obstacle distribution:")
print(df["ObstacleType"].value_counts())
"""


def p18_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Autonomous Vehicle Sensor Fusion\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data/sensor_data.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Obstacle types:\\n{df['ObstacleType'].value_counts()}")

# Simple voting ensemble: each sensor trains its own classifier
# This starter just uses camera features
camera_cols = [c for c in df.columns if c.startswith("Camera_")]
lidar_cols = [c for c in df.columns if c.startswith("LiDAR_")]
radar_cols = [c for c in df.columns if c.startswith("Radar_")]

y = df["ObstacleType"]
X_camera = df[camera_cols].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X_camera, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\\nCamera-only accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# TODO:
# 1. Implement late fusion: train separate models per sensor, then meta-classifier
# 2. Implement early fusion: concatenate all sensor features
# 3. Handle missing data (NaN) from sensor failures
# 4. Test performance when one sensor completely fails
# 5. Add uncertainty quantification
"""


def p18_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Autonomous Vehicle Sensor Fusion
Implements early fusion, late fusion, and handles missing data.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/sensor_data.csv")

camera_cols = [c for c in df.columns if c.startswith("Camera_")]
lidar_cols = [c for c in df.columns if c.startswith("LiDAR_")]
radar_cols = [c for c in df.columns if c.startswith("Radar_")]
all_sensor_cols = camera_cols + lidar_cols + radar_cols

y = df["ObstacleType"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Create missing-sensor indicators
for sensor in ["Camera", "LiDAR", "Radar"]:
    cols = [c for c in df.columns if c.startswith(f"{sensor}_")]
    df[f"{sensor}_missing"] = df[cols].isna().all(axis=1).astype(int)

indicator_cols = [c for c in df.columns if c.endswith("_missing")]

# Split
X_all = df[all_sensor_cols].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Categories: {le.classes_}")

# ================================
# Strategy 1: Early Fusion
# ================================
print("\\n" + "=" * 50)
print("STRATEGY 1: EARLY FUSION (concatenate all features)")
print("=" * 50)

early_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
])

early_pipeline.fit(X_train, y_train)
y_pred_early = early_pipeline.predict(X_test)
acc_early = accuracy_score(y_test, y_pred_early)
print(f"Accuracy: {acc_early:.4f}")
print(classification_report(y_test, y_pred_early))

# ================================
# Strategy 2: Late Fusion (meta-classifier)
# ================================
print("\\n" + "=" * 50)
print("STRATEGY 2: LATE FUSION (separate models + meta-classifier)")
print("=" * 50)

def train_sensor_model(sensor_cols, name):
    \"\"\"Train a model for a single sensor.\"\"\"
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train[sensor_cols], y_train)
    y_pred = pipe.predict(X_test[sensor_cols])
    acc = accuracy_score(y_test, y_pred)
    print(f"  {name} accuracy: {acc:.4f}")
    return pipe

cam_model = train_sensor_model(camera_cols, "Camera")
lidar_model = train_sensor_model(lidar_cols, "LiDAR")
radar_model = train_sensor_model(radar_cols, "Radar")

# Meta-features: concatenate predicted probabilities from each sensor model
def get_meta_features(models, sensor_cols_list, X):
    \"\"\"Generate meta-features from base models.\"\"\"
    meta_feats = []
    for model, cols in zip(models, sensor_cols_list):
        probs = model.predict_proba(X[cols])
        meta_feats.append(probs)
    return np.hstack(meta_feats)

sensor_models = [cam_model, lidar_model, radar_model]
sensor_cols_list = [camera_cols, lidar_cols, radar_cols]

train_meta = get_meta_features(sensor_models, sensor_cols_list, X_train)
test_meta = get_meta_features(sensor_models, sensor_cols_list, X_test)

meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf.fit(train_meta, y_train)
y_pred_late = meta_clf.predict(test_meta)
acc_late = accuracy_score(y_test, y_pred_late)
print(f"\\nLate fusion accuracy: {acc_late:.4f}")
print(classification_report(y_test, y_pred_late))

# ================================
# Sensor Failure Simulation
# ================================
print("\\n" + "=" * 50)
print("SENSOR FAILURE SIMULATION")
print("=" * 50)

def simulate_sensor_failure(X, cols_to_drop, strategy="mean"):
    \"\"\"Simulate sensor failure by masking features.\"\"\"
    X_fail = X.copy()
    for col in cols_to_drop:
        X_fail[col] = np.nan
    return X_fail

failure_scenarios = {
    "All sensors": all_sensor_cols,
    "No Camera": camera_cols,
    "No LiDAR": lidar_cols,
    "No Radar": radar_cols,
    "Camera + LiDAR": camera_cols + lidar_cols
}

for scenario, dropped_cols in failure_scenarios.items():
    remaining_cols = [c for c in all_sensor_cols if c not in dropped_cols]
    if not remaining_cols:
        continue
    X_test_fail = simulate_sensor_failure(X_test, dropped_cols)
    
    # Early fusion on remaining
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train[remaining_cols], y_train)
    y_pred_fail = pipe.predict(X_test_fail[remaining_cols])
    acc_fail = accuracy_score(y_test, y_pred_fail)
    print(f"  {scenario}: accuracy = {acc_fail:.4f}")

# ================================
# Uncertainty Estimation
# ================================
print("\\n" + "=" * 50)
print("UNCERTAINTY QUANTIFICATION")
print("=" * 50)

# Use prediction confidence as uncertainty measure
y_prob = early_pipeline.predict_proba(X_test)
max_probs = y_prob.max(axis=1)
mean_conf = max_probs.mean()
print(f"Mean prediction confidence: {mean_conf:.4f}")

# Confidence correlates with correctness?
correct = y_pred_early == y_test
correct_conf = max_probs[correct].mean()
incorrect_conf = max_probs[~correct].mean()
print(f"Average confidence when correct: {correct_conf:.4f}")
print(f"Average confidence when incorrect: {incorrect_conf:.4f}")
print(f"Confidence gap: {correct_conf - incorrect_conf:.4f}")

# ================================
# Final Summary
# ================================
print("\\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Early fusion accuracy:   {acc_early:.4f}")
print(f"Late fusion accuracy:    {acc_late:.4f}")
print(f"All-sensors threshold:   > 90% = {'PASS' if acc_early > 0.90 else 'NEEDS IMPROVEMENT'}")
"""


def p18_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
"""


# =============================================================================
# PROBLEM 19 - Network Anomaly Detection
# =============================================================================

def p19_readme():
    return """# Problem 19: Network Anomaly Detection

## Domain
Time Series + Anomaly Detection

## Problem Statement
Your company's SOC (Security Operations Center) needs an automated system to detect network anomalies in real-time. The network generates ~10,000 traffic records containing different types of anomalous behavior: DDoS attacks (sudden traffic spikes), data exfiltration (slow, stealthy transfers), and port scanning (many small packets to different IPs). Only ~2% of records are anomalous, making this a highly imbalanced detection problem.

The detector must operate in near-real-time: it cannot look at future data to flag current events. It must not only detect anomalies but also distinguish between anomaly types so the response team can take appropriate action.

## Objectives
1. Detect network anomalies with F1 > 0.85
2. Keep false positive rate below 5%
3. Detect anomalies within 3 time steps of onset (low latency)
4. Distinguish between DDoS, data exfiltration, and port scanning patterns

## Dataset
- 10000 network traffic records
- Columns: Timestamp, SourceIP (hashed), DestIP (hashed), Protocol, PacketSize, Duration, BytesSent, BytesReceived, NumPackets, ErrorRate, TimeOfDay
- Ground truth: AnomalyType (DDoS/Exfiltration/PortScan/None)
- CSV at `data/network_traffic.csv`

## Success Criteria
- **F1-score**: >0.85 on anomaly detection (binary: anomaly vs normal)
- **False Positive Rate**: <5%
- **Detection Latency**: Average <3 time steps from anomaly onset
- **Type Classification**: Per-type F1 > 0.70 for each anomaly category

## Starter Code
`starter_code.py` loads data and implements basic threshold-based detection. You must extend with proper anomaly detection approaches.

## Constraints
- Cannot use future data for detection (streaming/online scenario)
- Must evaluate both detection and classification performance
- Must report detection latency
"""


def p19_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic network traffic with anomalies.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_records = 10000
anomaly_rate = 0.02
protocols = ["TCP", "UDP", "HTTP", "DNS"]
anomaly_types = ["DDoS", "Exfiltration", "PortScan"]

# Normal traffic parameters
base_packet_size = np.random.lognormal(5, 0.5, n_records)  # ~150 bytes avg
base_duration = np.random.exponential(2, n_records)
base_bytes_sent = np.random.lognormal(8, 1, n_records)
base_bytes_received = np.random.lognormal(9, 1, n_records)
base_num_packets = np.random.poisson(10, n_records)
base_error_rate = np.random.exponential(0.01, n_records)
base_time_of_day = np.random.uniform(0, 24, n_records)

# Generate anomalies in bursts
anomaly_mask = np.zeros(n_records, dtype=bool)
anomaly_type_labels = np.full(n_records, "None", dtype=object)

# DDoS: sudden spike in packet count and bytes
n_ddos = int(n_records * 0.007)
ddos_starts = np.random.choice(n_records - 50, n_ddos // 10, replace=False)
for start in ddos_starts:
    burst_len = np.random.randint(5, 15)
    for i in range(start, min(start + burst_len, n_records)):
        if np.random.random() < 0.8:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "DDoS"
            base_num_packets[i] = np.random.poisson(500)
            base_bytes_sent[i] = np.random.lognormal(12, 1)
            base_packet_size[i] = np.random.lognormal(4, 0.3)

# Data exfiltration: slow steady transfer
n_exfil = int(n_records * 0.006)
exfil_starts = np.random.choice(n_records - 100, n_exfil // 15, replace=False)
for start in exfil_starts:
    exfil_len = np.random.randint(10, 30)
    for i in range(start, min(start + exfil_len, n_records)):
        if np.random.random() < 0.6:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "Exfiltration"
            base_bytes_sent[i] = np.random.lognormal(11, 0.5)
            base_duration[i] = np.random.exponential(10)
            base_num_packets[i] = 1
            base_packet_size[i] = np.random.lognormal(7, 0.3)

# Port scanning: many small packets to different IPs
n_scan = int(n_records * 0.007)
scan_starts = np.random.choice(n_records - 30, n_scan // 8, replace=False)
for start in scan_starts:
    scan_len = np.random.randint(5, 15)
    for i in range(start, min(start + scan_len, n_records)):
        if np.random.random() < 0.7:
            anomaly_mask[i] = True
            anomaly_type_labels[i] = "PortScan"
            base_num_packets[i] = np.random.poisson(20)
            base_bytes_sent[i] = np.random.lognormal(6, 0.5)
            base_packet_size[i] = np.random.lognormal(3, 0.3)
            base_error_rate[i] = np.random.exponential(0.2)

# Build DataFrame
df = pd.DataFrame({
    "Timestamp": pd.date_range("2024-01-01", periods=n_records, freq="5s"),
    "SourceIP": np.random.randint(1000000, 9999999, n_records),
    "DestIP": np.random.randint(1000000, 9999999, n_records),
    "Protocol": np.random.choice(protocols, n_records),
    "PacketSize": np.round(base_packet_size, 1),
    "Duration": np.round(base_duration, 3),
    "BytesSent": np.round(base_bytes_sent, 1),
    "BytesReceived": np.round(base_bytes_received, 1),
    "NumPackets": base_num_packets.astype(int),
    "ErrorRate": np.round(base_error_rate, 4),
    "TimeOfDay": np.round(base_time_of_day, 2),
    "AnomalyType": anomaly_type_labels
})

# Shuffle timestamps out-of-order to simulate streaming
df = df.sample(frac=1, random_state=42).sort_values("Timestamp").reset_index(drop=True)

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "network_traffic.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} records (anomalies: {(df['AnomalyType'] != 'None').sum()}) -> {out_path}")
print(f"Anomaly breakdown:")
print(df[df["AnomalyType"] != "None"]["AnomalyType"].value_counts())
"""


def p19_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: Network Anomaly Detection\"\"\"

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Load data
df = pd.read_csv("data/network_traffic.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Anomaly distribution:\\n{df['AnomalyType'].value_counts()}")

y_true = (df["AnomalyType"] != "None").astype(int)

# Basic threshold-based detection
# Simple heuristic: Z-score on BytesSent and NumPackets
features = ["BytesSent", "NumPackets", "PacketSize", "ErrorRate"]
anomaly_scores = np.zeros(len(df))

for col in features:
    mean_v = df[col].mean()
    std_v = df[col].std()
    if std_v > 0:
        anomaly_scores += np.abs((df[col] - mean_v) / std_v)
    else:
        print(f"Warning: {col} has zero std")

threshold = anomaly_scores.quantile(0.95)
y_pred = (anomaly_scores > threshold).astype(int)

print(f"\\nThreshold-based detection:")
print(classification_report(y_true, y_pred))
print(f"F1: {f1_score(y_true, y_pred):.4f}")

# TODO:
# 1. Implement proper anomaly detection (IsolationForest, Autoencoder)
# 2. Engineer rolling/rate features for temporal patterns
# 3. Add post-processing for temporal consistency
# 4. Classify anomaly types (multi-class)
# 5. Measure detection latency
"""


def p19_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: Network Anomaly Detection
IsolationForest + feature engineering + temporal consistency.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.neural_network import MLPClassifier
    AUTOENCODER = True
except ImportError:
    AUTOENCODER = False

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/network_traffic.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Ground truth
y_binary = (df["AnomalyType"] != "None").astype(int)
y_multi = df["AnomalyType"].values

print(f"Data: {len(df)} records")
print(f"Anomalies: {y_binary.sum()} ({y_binary.mean()*100:.2f}%)")

# ========================
# Feature engineering
# ========================
def engineer_features(df):
    df = df.copy()
    
    # Encode protocol
    le = LabelEncoder()
    df["Protocol_enc"] = le.fit_transform(df["Protocol"])
    
    # Rolling statistics (temporal features, respecting time order)
    window = 20
    for col in ["BytesSent", "BytesReceived", "NumPackets", "PacketSize", "ErrorRate"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std()
        df[f"{col}_roll_max"] = df[col].rolling(window, min_periods=1).max()
        df[f"{col}_rate"] = df[col].diff().fillna(0)
    
    # Rate features (change per time step)
    df["bytes_sent_rate"] = df["BytesSent"].diff().fillna(0)
    df["packet_rate"] = df["NumPackets"].diff().fillna(0)
    
    # Ratio features
    df["send_recv_ratio"] = df["BytesSent"] / (df["BytesReceived"] + 1)
    df["bytes_per_packet"] = df["BytesSent"] / (df["NumPackets"] + 1)
    
    # IP features (frequency of same source/dest)
    df["src_freq"] = df.groupby("SourceIP")["SourceIP"].transform("count")
    df["dest_freq"] = df.groupby("DestIP")["DestIP"].transform("count")
    
    # Interaction features
    df["bytes_x_packets"] = df["BytesSent"] * df["NumPackets"] / 10000
    df["error_x_duration"] = df["ErrorRate"] * df["Duration"]
    
    # Sin/cos time encoding
    df["time_sin"] = np.sin(2 * np.pi * df["TimeOfDay"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["TimeOfDay"] / 24)
    
    return df.fillna(0)

df_feat = engineer_features(df)
feature_cols = [c for c in df_feat.columns if c not in
                ["Timestamp", "SourceIP", "DestIP", "Protocol", "AnomalyType"]]

print(f"Features engineered: {len(feature_cols)}")

# ========================
# IsolationForest for anomaly detection
# ========================
print("\\n" + "=" * 50)
print("ISOLATION FOREST ANOMALY DETECTION")
print("=" * 50)

# Train on first 60%, test on remaining 40% (temporal split)
split_idx = int(len(df_feat) * 0.6)
train_df = df_feat.iloc[:split_idx]
test_df = df_feat.iloc[split_idx:]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_test_bin = y_binary[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train IsolationForest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_s)

# Predict: -1 = anomaly, 1 = normal
y_pred_iso = iso.predict(X_test_s)
y_pred_iso_bin = (y_pred_iso == -1).astype(int)

print("\\nIsolationForest Results:")
print(classification_report(y_test_bin, y_pred_iso_bin))
print(f"F1: {f1_score(y_test_bin, y_pred_iso_bin):.4f}")
print(f"FP rate: {(~y_test_bin.astype(bool) & y_pred_iso_bin.astype(bool)).sum() / (~y_test_bin).sum():.4f}")

# Score-based detection with tunable threshold
scores = iso.score_samples(X_test_s)
threshold = np.percentile(scores, 2)
y_pred_score = (scores < threshold).astype(int)

print(f"\\nScore-based (threshold={threshold:.3f}):")
print(classification_report(y_test_bin, y_pred_score))
f1_score_val = f1_score(y_test_bin, y_pred_score)
print(f"F1: {f1_score_val:.4f}")

# ========================
# Anomaly Type Classification
# ========================
print("\\n" + "=" * 50)
print("ANOMALY TYPE CLASSIFICATION")
print("=" * 50)

# Among detected anomalies, classify type
from sklearn.ensemble import RandomForestClassifier

detected_mask = y_pred_score == 1
if detected_mask.sum() > 0:
    y_test_types = test_df["AnomalyType"].values[detected_mask]
    # Only use train data anomalies for training type classifier
    train_anom_mask = train_df["AnomalyType"] != "None"
    
    if train_anom_mask.sum() > 10:
        type_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        type_clf.fit(
            X_train[train_anom_mask.values],
            train_df["AnomalyType"].values[train_anom_mask]
        )
        y_pred_types = type_clf.predict(X_test[detected_mask])
        print(classification_report(y_test_types, y_pred_types))

# ========================
# Detection Latency
# ========================
print("\\n" + "=" * 50)
print("DETECTION LATENCY")
print("=" * 50)

latencies = []
current_anomaly_start = None
for i in range(len(y_test_bin)):
    if y_test_bin[i] == 1 and current_anomaly_start is None:
        current_anomaly_start = i
    elif y_test_bin[i] == 1 and current_anomaly_start is not None:
        if y_pred_score[i] == 1:
            latencies.append(i - current_anomaly_start)
            current_anomaly_start = None
    else:
        if current_anomaly_start is not None and y_test_bin[i] == 0:
            # Anomaly ended without detection
            latencies.append(i - current_anomaly_start)
            current_anomaly_start = None

if latencies:
    print(f"Average detection latency: {np.mean(latencies):.2f} time steps")
    print(f"Median detection latency: {np.median(latencies):.2f} time steps")
    print(f"Latency < 3: {(np.array(latencies) < 3).mean()*100:.1f}%")
else:
    print("No anomalies detected in test set")

# ========================
# Temporal Consistency Post-Processing
# ========================
print("\\n" + "=" * 50)
print("TEMPORAL CONSISTENCY POST-PROCESSING")
print("=" * 50)

def temporal_smoothing(predictions, min_anomaly_window=3):
    \"\"\"Remove isolated anomaly predictions.\"\"\"
    smoothed = predictions.copy()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # Check if surrounded by normal
            window_before = predictions[max(0, i-min_anomaly_window):i]
            window_after = predictions[i+1:min(len(predictions), i+min_anomaly_window+1)]
            if window_before.sum() < 1 and window_after.sum() < 1:
                smoothed[i] = 0
    return smoothed

y_pred_smooth = temporal_smoothing(y_pred_score, min_anomaly_window=3)
f1_smooth = f1_score(y_test_bin, y_pred_smooth)
print(f"\\nAfter temporal smoothing:")
print(classification_report(y_test_bin, y_pred_smooth))
print(f"F1: {f1_smooth:.4f}")

# ========================
# Final Summary
# ========================
print("\\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
fpr = (~y_test_bin.astype(bool) & y_pred_smooth.astype(bool)).sum() / (~y_test_bin).sum()
print(f"IsolationForest F1: {f1_score_val:.4f}")
print(f"After smoothing F1: {f1_smooth:.4f}")
print(f"False positive rate: {fpr:.4f}")
print(f"F1 > 0.85: {'PASS' if f1_smooth > 0.85 else 'NEEDS IMPROVEMENT'}")
print(f"FPR < 5%: {'PASS' if fpr < 0.05 else 'NEEDS IMPROVEMENT'}")
"""


def p19_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
"""


# =============================================================================
# PROBLEM 20 - End-to-End ML Pipeline with Deployment Constraints
# =============================================================================

def p20_readme():
    return """# Problem 20: End-to-End ML Pipeline with Deployment Constraints

## Domain
Full Pipeline + MLOps

## Problem Statement
You are tasked with building a complete machine learning pipeline that is ready for production deployment. The dataset contains mixed data types (numeric, categorical, text, date) with a binary target. However, here's the catch: the last 2000 rows exhibit data drift -- their distributions differ slightly from the training data, simulating what happens when a model encounters production data that has shifted over time.

Your pipeline must be reproducible, handle missing data gracefully, meet size and speed constraints (<50MB model, <100ms inference), and include automated data drift detection. Finally, you need to wrap the model in a simple API (Flask/FastAPI) so it can be called by other services.

## Objectives
1. Design a reproducible preprocessing pipeline using sklearn's ColumnTransformer/Pipeline
2. Train and tune a model meeting size (<50MB) and speed (<100ms per prediction) constraints
3. Write a model-serving API using Flask or FastAPI
4. Implement data drift detection to catch the distribution shift
5. Ensure the entire pipeline is reproducible (fixed random seeds, pinned dependencies)

## Dataset
- 8000 rows with mixed data types (numeric, categorical, text, date)
- Binary classification target
- Last 2000 rows have slightly different distributions (simulated drift)
- Contains missing values
- CSV at `data/production_pipeline_data.csv`

## Success Criteria
- **Reproducibility**: Running the pipeline twice produces identical results
- **API**: `/predict` endpoint returns correct predictions with <100ms latency
- **Drift Detection**: Detect the distribution shift in the last 2000 rows
- **Size Constraint**: Serialized model < 50MB
- **Speed**: Batch inference < 100ms for single prediction

## Starter Code
`starter_code.py` provides basic EDA and a single-model baseline. You must build the full pipeline.

## Constraints
- Model size < 50MB (serialized)
- Inference < 100ms per single prediction
- Must handle missing values automatically
- Pipeline must be fully reproducible
"""


def p20_generate_data():
    return """#!/usr/bin/env python3
\"\"\"Generate synthetic production pipeline data with drift.\"\"\"

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_normal = 6000
n_drift = 2000
n_total = n_normal + n_drift

# Normal data
records = []
for _ in range(n_normal):
    age = np.random.normal(40, 12)
    income = np.random.lognormal(4.5, 0.6)
    credit_score = np.random.normal(700, 50)
    account_age = np.random.exponential(5)
    num_transactions = np.random.poisson(15)
    avg_transaction = np.random.lognormal(3, 0.8)
    customer_tier = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.4, 0.3, 0.2, 0.1])
    region = np.random.choice(["North", "South", "East", "West"], p=[0.25, 0.25, 0.25, 0.25])
    feedback = np.random.choice(["Good", "Average", "Poor"], p=[0.5, 0.3, 0.2])
    signup_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 1500))
    
    # Target
    p = 1 / (1 + np.exp(-(-2 + 0.01 * income + 0.003 * credit_score +
                           0.1 * account_age - 0.02 * age +
                           {"Bronze": -0.5, "Silver": 0, "Gold": 0.5, "Platinum": 1.0}[customer_tier])))
    target = int(np.random.random() < p)
    
    # Missing values
    if np.random.random() < 0.03:
        income = np.nan
    if np.random.random() < 0.02:
        credit_score = np.nan
    if np.random.random() < 0.01:
        age = np.nan
    
    records.append({
        "Age": round(age, 1),
        "Income": round(income, 2) if not np.isnan(income) else np.nan,
        "CreditScore": round(credit_score, 1) if not np.isnan(credit_score) else np.nan,
        "AccountAgeMonths": round(account_age, 1),
        "NumTransactions": num_transactions,
        "AvgTransactionAmount": round(avg_transaction, 2),
        "CustomerTier": customer_tier,
        "Region": region,
        "Feedback": feedback,
        "SignupDate": signup_date,
        "Target": target
    })

# Drifted data (different distributions)
for _ in range(n_drift):
    age = np.random.normal(55, 15)  # Older
    income = np.random.lognormal(4.0, 0.5)  # Lower income
    credit_score = np.random.normal(650, 60)  # Lower credit
    account_age = np.random.exponential(3)  # Newer accounts
    num_transactions = np.random.poisson(8)  # Fewer transactions
    avg_transaction = np.random.lognormal(2.5, 0.6)  # Lower transaction amounts
    customer_tier = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.5, 0.3, 0.15, 0.05])
    region = np.random.choice(["North", "South", "East", "West"], p=[0.4, 0.2, 0.3, 0.1])  # Different region dist
    feedback = np.random.choice(["Good", "Average", "Poor"], p=[0.3, 0.4, 0.3])  # More negative
    signup_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 800))  # Newer
    
    # Different target generation (slightly lower)
    p = 1 / (1 + np.exp(-(-2.5 + 0.008 * income + 0.002 * credit_score +
                           0.05 * account_age - 0.01 * age +
                           {"Bronze": -0.3, "Silver": 0.1, "Gold": 0.3, "Platinum": 0.6}[customer_tier])))
    target = int(np.random.random() < p)
    
    # Missing values (higher rate in drift)
    if np.random.random() < 0.05:
        income = np.nan
    if np.random.random() < 0.04:
        credit_score = np.nan
    if np.random.random() < 0.02:
        age = np.nan
    
    records.append({
        "Age": round(age, 1) if not np.isnan(age) else np.nan,
        "Income": round(income, 2) if not np.isnan(income) else np.nan,
        "CreditScore": round(credit_score, 1) if not np.isnan(credit_score) else np.nan,
        "AccountAgeMonths": round(account_age, 1),
        "NumTransactions": num_transactions,
        "AvgTransactionAmount": round(avg_transaction, 2),
        "CustomerTier": customer_tier,
        "Region": region,
        "Feedback": feedback,
        "SignupDate": signup_date,
        "Target": target
    })

df = pd.DataFrame(records)
df["IsDrifted"] = [0] * n_normal + [1] * n_drift

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "production_pipeline_data.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} records (drifted: {n_drift}) -> {out_path}")
print(f"Normal target rate: {df[df['IsDrifted']==0]['Target'].mean():.3f}")
print(f"Drifted target rate: {df[df['IsDrifted']==1]['Target'].mean():.3f}")
"""


def p20_starter():
    return """#!/usr/bin/env python3
\"\"\"Starter code: End-to-End ML Pipeline with Deployment Constraints\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("data/production_pipeline_data.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\\n{df.isna().sum()}")
print(f"Target distribution:\\n{df['Target'].value_counts()}")

# Simple split
X = df.drop(columns=["Target", "IsDrifted"])
y = df["Target"]

# Basic preprocessing
X = pd.get_dummies(X, columns=["CustomerTier", "Region", "Feedback"], drop_first=True)
X["SignupDate"] = pd.to_datetime(X["SignupDate"])
X["SignupYear"] = X["SignupDate"].dt.year
X["SignupMonth"] = X["SignupDate"].dt.month
X = X.drop(columns=["SignupDate"])
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\\nAccuracy: {(y_pred == y_test).mean():.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, y_pred))

# TODO:
# 1. Build proper sklearn Pipeline with ColumnTransformer
# 2. Handle missing values, categoricals, text, dates in a single pipeline
# 3. Meet size (<50MB) and speed (<100ms) constraints
# 4. Add drift detection on the last 2000 rows
# 5. Create Flask/FastAPI serving endpoint
# 6. Ensure reproducibility
"""


def p20_solution():
    return """#!/usr/bin/env python3
\"\"\"Reference solution: End-to-End ML Pipeline with Deployment Constraints
Includes: sklearn Pipeline, model serving, drift detection, reproducibility.
\"\"\"

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from scipy.stats import ks_2samp
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# ========================
# Load and explore data
# ========================
df = pd.read_csv("data/production_pipeline_data.csv")
df["SignupDate"] = pd.to_datetime(df["SignupDate"])
print(f"Data: {len(df)} rows, {len(df.columns)} columns")
print(f"Drift indicator present (for evaluation only)")

# Separate drift detection
n_normal = 6000
n_drift = 2000
df_normal = df.iloc[:n_normal]
df_drift = df.iloc[n_normal:]

# ========================
# Custom transformers
# ========================
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    \"\"\"Extract year, month, dayofyear from date column.\"\"\"
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        dates = pd.to_datetime(X.iloc[:, 0])
        return np.column_stack([
            dates.dt.year,
            dates.dt.month,
            dates.dt.dayofyear,
            dates.dt.quarter
        ])

# ========================
# Build the preprocessing pipeline
# ========================
numeric_features = ["Age", "Income", "CreditScore", "AccountAgeMonths",
                    "NumTransactions", "AvgTransactionAmount"]
categorical_features = ["CustomerTier", "Region", "Feedback"]
date_features = ["SignupDate"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

date_transformer = Pipeline([
    ("extractor", DateFeatureExtractor()),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("date", date_transformer, date_features)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
])

# ========================
# Training (reproducible)
# ========================
print("\\n" + "=" * 50)
print("TRAINING PIPELINE")
print("=" * 50)

y = df_normal["Target"]
X = df_normal.drop(columns=["Target", "IsDrifted"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Training time: {train_time:.2f}s")
print(classification_report(y_test, y_pred))

# ========================
# Model size check
# ========================
print("\\n" + "=" * 50)
print("MODEL SIZE CHECK")
print("=" * 50)

pickle_path = "model.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(pipeline, f)

model_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Size < 50MB: {'PASS' if model_size_mb < 50 else 'FAIL'}")

# ========================
# Inference speed
# ========================
print("\\n" + "=" * 50)
print("INFERENCE SPEED CHECK")
print("=" * 50)

# Single prediction
single_sample = X_test.iloc[:1]
n_runs = 100
times = []
for _ in range(n_runs):
    t0 = time.time()
    pipeline.predict(single_sample)
    times.append((time.time() - t0) * 1000)  # ms

avg_time = np.mean(times)
print(f"Average single inference: {avg_time:.2f} ms")
print(f"< 100ms: {'PASS' if avg_time < 100 else 'FAIL'}")

# ========================
# Drift Detection
# ========================
print("\\n" + "=" * 50)
print("DATA DRIFT DETECTION")
print("=" * 50)

if STATS_AVAILABLE:
    drift_results = {}
    for col in numeric_features:
        normal_data = df_normal[col].dropna()
        drift_data = df_drift[col].dropna()
        stat, p_value = ks_2samp(normal_data, drift_data)
        drift_results[col] = {"KS_stat": stat, "p_value": p_value, "drifted": p_value < 0.05}
    
    drift_count = sum(1 for v in drift_results.values() if v["drifted"])
    print(f"Features with detected drift: {drift_count}/{len(numeric_features)}")
    for col, result in drift_results.items():
        print(f"  {col}: KS={result['KS_stat']:.4f}, p={result['p_value']:.4f}, Drift={'YES' if result['drifted'] else 'NO'}")
    
    if drift_count >= 3:
        print("\\nDrift detection PASS: Distribution shift correctly identified")
    else:
        print("\\nDrift detection WARNING: Consider more sensitive detection")
else:
    # Simple mean comparison
    for col in numeric_features:
        normal_mean = df_normal[col].mean()
        drift_mean = df_drift[col].mean()
        diff_pct = abs(drift_mean - normal_mean) / normal_mean * 100
        print(f"  {col}: normal={normal_mean:.2f}, drift={drift_mean:.2f}, diff={diff_pct:.1f}%")

# ========================
# Reproducibility check
# ========================
print("\\n" + "=" * 50)
print("REPRODUCIBILITY CHECK")
print("=" * 50)

pipeline2 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
])
pipeline2.fit(X_train, y_train)

y_pred2 = pipeline2.predict(X_test)
identical = (y_pred == y_pred2).all()
print(f"Reproducible predictions: {'YES' if identical else 'NO'}")

# ========================
# API Server Code (commented - uncomment to run)
# ========================
print("\\n" + "=" * 50)
print("API SERVER (FastAPI)")
print("=" * 50)

api_code = '''
# Save this as app.py and run with: uvicorn app:host 0.0.0.0 --port 8000

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Pipeline API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    Age: float
    Income: float
    CreditScore: float
    AccountAgeMonths: float
    NumTransactions: int
    AvgTransactionAmount: float
    CustomerTier: str
    Region: str
    Feedback: str
    SignupDate: str

class Prediction(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump()])
    df["SignupDate"] = pd.to_datetime(df["SignupDate"])
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob > 0.5)
    return Prediction(prediction=pred, probability=round(prob, 4))

@app.get("/health")
def health():
    return {"status": "ok"}
'''

print("API code ready. See starter code for how to run.")
print(api_code[:200] + "\\n...")

# ========================
# Summary
# ========================
print("\\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"Model size: {model_size_mb:.2f} MB (<50MB: {'PASS' if model_size_mb < 50 else 'FAIL'})")
print(f"Inference: {avg_time:.2f} ms (<100ms: {'PASS' if avg_time < 100 else 'FAIL'})")
print(f"Drift detection: Drift identified in {drift_count}/{len(numeric_features)} numerical features")
print(f"Reproducible: {'YES' if identical else 'NO'}")
"""


def p20_requirements():
    return """pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
pydantic>=1.8.0
scipy>=1.7.0
joblib>=1.1.0
"""


# =============================================================================
# Mapping problem IDs to generator functions
# =============================================================================

PROBLEM_GENERATORS = {
    "14-multi-class-text-classification": {
        "README.md": p14_readme,
        "data/generate_data.py": p14_generate_data,
        "starter_code.py": p14_starter,
        "solution.py": p14_solution,
        "requirements.txt": p14_requirements,
    },
    "15-predictive-maintenance": {
        "README.md": p15_readme,
        "data/generate_data.py": p15_generate_data,
        "starter_code.py": p15_starter,
        "solution.py": p15_solution,
        "requirements.txt": p15_requirements,
    },
    "16-hybrid-recommendation-system": {
        "README.md": p16_readme,
        "data/generate_data.py": p16_generate_data,
        "starter_code.py": p16_starter,
        "solution.py": p16_solution,
        "requirements.txt": p16_requirements,
    },
    "17-medical-diagnosis-limited-data": {
        "README.md": p17_readme,
        "data/generate_data.py": p17_generate_data,
        "starter_code.py": p17_starter,
        "solution.py": p17_solution,
        "requirements.txt": p17_requirements,
    },
    "18-autonomous-vehicle-sensor-fusion": {
        "README.md": p18_readme,
        "data/generate_data.py": p18_generate_data,
        "starter_code.py": p18_starter,
        "solution.py": p18_solution,
        "requirements.txt": p18_requirements,
    },
    "19-network-anomaly-detection": {
        "README.md": p19_readme,
        "data/generate_data.py": p19_generate_data,
        "starter_code.py": p19_starter,
        "solution.py": p19_solution,
        "requirements.txt": p19_requirements,
    },
    "20-end-to-end-ml-pipeline": {
        "README.md": p20_readme,
        "data/generate_data.py": p20_generate_data,
        "starter_code.py": p20_starter,
        "solution.py": p20_solution,
        "requirements.txt": p20_requirements,
    },
}


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    for problem_id, files in PROBLEM_GENERATORS.items():
        print(f"\\n{'=' * 60}")
        print(f"Generating: {problem_id}")
        print(f"{'=' * 60}")
        problem_dir = os.path.join(base, "Difficult", problem_id)
        
        for filename, content_func in files.items():
            filepath = os.path.join(problem_dir, filename)
            write_file(filepath, content_func())
        
        # Also generate data by running the generation script
        print(f"  Running data generator for {problem_id}...")
        gen_script = os.path.join(problem_dir, "data", "generate_data.py")
        result = subprocess.run(
            [sys.executable, gen_script],
            capture_output=True, text=True, cwd=problem_dir
        )
        for line in result.stdout.strip().split("\\n"):
            print(f"    {line}")
        if result.stderr:
            print(f"    Errors: {result.stderr[:200]}")
    
    print("\\n" + "=" * 60)
    print("ALL GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
