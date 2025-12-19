# 🛒 Real-Time Analysis & Sentiment Detection of Amazon Electronics Reviews

**Authors:**

* **Anand Nareshkumar Patel**, Department of Computer Science, Rutgers University — *[ap3085@scarletmail.rutgers.edu](mailto:ap3085@scarletmail.rutgers.edu)*
* **Anish Vishnu Shirodkar**, Department of Computer Science, Rutgers University — *[sa2792@scarletmail.rutgers.edu](mailto:sa2792@scarletmail.rutgers.edu)*
* **Santanu Agarwal**, Department of Computer Science, Rutgers University — *[avs181@scarletmail.rutgers.edu](mailto:avs181@scarletmail.rutgers.edu)*

📄 **Paper (NeurIPS-style write-up):** *Included in repository*
🎥 **Video Demo:** [https://drive.google.com/drive/folders/1pZVumiUZ8kDg64_L9Nos3egtc8w0X0O4](https://drive.google.com/drive/folders/1pZVumiUZ8kDg64_L9Nos3egtc8w0X0O4)
💻 **Code Repository:** [https://github.com/anand25116/Real_time_analysis_and_Sentiment_detection](https://github.com/anand25116/Real_time_analysis_and_Sentiment_detection)

---

## 📌 Overview

This project implements an **end-to-end sentiment classification pipeline** on large-scale
**Amazon Electronics reviews**, using:

### 🔹 Baseline Model

* **TF-IDF + Logistic Regression**

### 🔹 Transformer Models

* **DistilBERT**
* **RoBERTa**

We compare:

* Prediction accuracy
* Computational efficiency
* Contextual understanding
* Interpretability trade-offs

In addition, we extract **topics from positive vs negative reviews** and build a **Streamlit real-time review simulator** that streams reviews, visualizes model predictions, and shows product information.

---

## 🎯 Motivation

* Amazon review volume makes **manual trend analysis impossible**
* Sellers need **fast detection of dissatisfaction spikes**
* Users care about aspects like **battery life, durability, and performance**
* Transformers capture **context, negation, sarcasm**, unlike Bag-of-Words

Our model:

* Classifies reviews into **negative (0), neutral (1), positive (2)**
* Shows major **product aspect mentions**
* Streams **live review sentiment**

---

## 📂 Dataset Summary

### 🟦 Source

* **Amazon Reviews 2023 — McAuley-Lab (Electronics subset)**
* ~100,000 reviews sampled using chunk-loading for memory efficiency

Each record includes:

* review text
* star rating (1–5)
* timestamp
* product identifier

### 🟩 Sentiment Mapping

| Star Rating | Sentiment    |
| ----------- | ------------ |
| ⭐ 1–2       | Negative (0) |
| ⭐ 3         | Neutral (1)  |
| ⭐ 4–5       | Positive (2) |

### 🟨 Supplementary Dataset (for Streamlit UI)

* **Kaggle Amazon Sales**
* Images, product names, pricing

Not used in training — used for **visual context**.

---

## 🧹 Data Cleaning Pipeline

* Load via chunked `pandas.read_json(..., chunksize=50000)`
* Lower-casing
* Strip HTML tags
* Remove URLs
* Keep only alphabetic content
* Filter short reviews
* Convert UNIX timestamps
* Compute review length feature
* Save **electronics_reviews_clean.csv**

---

## 🧪 Baseline Model: TF-IDF + Logistic Regression

* `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`
* Stratified train-test split (80/20)
* `LogisticRegression(max_iter=1000, solver="lbfgs")`

Artifacts saved:

```
models/tfidf.pkl
models/logreg.pkl
models/baseline_config.pkl
```

---

## 🤖 Transformer Training

Both models fine-tuned using Hugging Face `Trainer`.

### 🟦 DistilBERT

* `distilbert-base-uncased`
* `num_labels=3`
* Batch-size=16
* Epochs=2
* `max_length=256`

Saved to:

```
models/bert_sentiment/
```

### 🟩 RoBERTa

* `roberta-base`
* Batch-size=8
* Epochs=2

Saved to:

```
models/roberta_sentiment/
```

---

## 📊 Model Performance

A bar chart compares:

* Logistic Regression
* DistilBERT
* RoBERTa

Transformers outperform baseline in accuracy and contextual handling — consistent with literature.

---

## 🔍 Aspect-Level Insights

Using spaCy noun phrase extraction:

* Extract most common nouns in top 500 positive reviews
* Extract most common nouns in top 500 negative reviews

Provides **business intelligence** (battery, sound, cable, warranty, etc.).

---

## 📈 Visualizations

* Sentiment class distribution
* Model accuracy bars
* Top aspects word frequency
* Real-time streaming UI

---

## 🖥️ Streamlit Real-Time Sentiment Simulator

Features:

* Upload CSV
* Product search (e.g., “iPhone 15”)
* Display product image
* Stream N reviews with configurable interval
* Predictions from:

  * BERT pipeline
  * RoBERTa pipeline
  * TF-IDF Logistic baseline

Color-coded:

* 🟢 Positive
* 🟡 Neutral
* 🔴 Negative

---

## 🏗️ System Architecture

```
Data Collection → Cleaning → Labeling → Train Test Split
     ↓
Baseline Model (TF-IDF + LR)
     ↓
Transformer Fine-Tuning (DistilBERT & RoBERTa)
     ↓
Evaluation → Aspect Extraction → Real-Time Web App
```

---

## 🛠️ Installation

```bash
git clone https://github.com/anand25116/Real_time_analysis_and_Sentiment_detection
cd Real_time_analysis_and_Sentiment_detection
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
data/
    Electronics.jsonl
    electronics_reviews_clean.csv
models/
    tfidf.pkl
    logreg.pkl
    bert_sentiment/
    roberta_sentiment/
notebooks/
streamlit_app/
results/
```

---

## 🧬 Key Findings

* Logistic Regression is efficient but shallow
* Transformers excel when subtle context matters
* RoBERTa > DistilBERT > Logistic Regression
* Topic extraction adds **explainability**

---

## 🧭 Applications

| Domain           | Benefit                               |
| ---------------- | ------------------------------------- |
| E-commerce       | Detect failing products early         |
| Marketing        | Track user response live              |
| Customer Support | Prioritize negative surge             |
| Research         | Classical vs Transformer benchmarking |

---

## 🚀 Future Enhancements

* Sliding window anomaly detection
* Time-series sentiment drift
* Prompt-based LLM evaluation
* Multi-aspect sentiment tagging
* Deploy app on Hugging Face Spaces
* Kafka streaming integration

---

## 📝 License

MIT — open for academic + commercial experimentation.

---

## ⭐ Support

If this work helped you:
**give the repo a star ⭐** — it motivates more research and open-source releases.

---
