# Problem 20: End-to-End ML Pipeline with Deployment Constraints

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
