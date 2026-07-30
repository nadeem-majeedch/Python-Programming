#!/usr/bin/env python3
"""Generate synthetic multi-class text dataset with overlapping vocabulary."""

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
