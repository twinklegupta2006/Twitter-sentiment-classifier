# Twitter Sentiment Classifier

A machine learning project that classifies tweets as Positive, Negative, or Neutral, with an interactive Streamlit app for trying it out.

## Overview

This project classifies tweets into 3 sentiment classes: Positive, Negative, Neutral.

Three models were trained and compared on the same data split:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- LSTM + GloVe

TF-IDF + Linear SVM is the final model used in the app.

## Results

On the held-out test set:

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 85.53% | 85.34% |
| TF-IDF + Linear SVM | 90.72% | 90.64% |
| LSTM + GloVe | 90.67% | 90.62% |

## Streamlit Demo

The app supports:
- single-text prediction
- batch prediction (multiple lines of text at once)

Run it locally:

```bash
streamlit run app.py
```

## Project Structure

- `app.py` — Streamlit application
- `src/` — ML and evaluation code (data loading, preprocessing, models, inference)
- `scripts/` — training, evaluation, benchmarking, and prediction scripts
- `artifacts/` — saved model artifacts used by the app
- `configs/` — configuration (`config.yaml`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The data comes from the Kaggle ["Twitter Entity Sentiment Analysis"](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset.

It's not included in this repo. You only need it if you want to retrain the models or reproduce the benchmark/evaluation scripts — not to run the Streamlit demo. Download `twitter_training.csv` and place it at `data/twitter_training.csv`.

