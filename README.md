# Zepto AI/ML Engineer Capstone Project

Welcome to the Zepto Analytics Guild capstone repository! This project integrates a full-stack data engineering pipeline, an advanced exploratory data analysis and predictive modeling pipeline, and a grounded GenAI policy support assistant into a single cohesive codebase.

---

## 📂 Repository Structure

- `data_pipeline/` — Scrapes live catalog data from `books.toscrape.com`, cleans fields, applies fixed-rate currency conversion, and loads it into a normalized SQLite database queried via SQL and pandas.
- `analytics/` — Profiles the Titanic dataset, builds a visual data story, implements leakage-free scikit-learn classification pipelines, handles class imbalance via SMOTE, evaluates regression, and saves a serialized `joblib` artifact.
- `support_assistant/` — Implements a grounded Retrieval-Augmented Generation (RAG) policy assistant for Zepto's internal policies using ChromaDB, embeddings, and LangGraph workflow routing.

---

## 🛠️ Project Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rajiycareer/zepto-ai-capstone.git](https://github.com/rajiycareer/zepto-ai-capstone.git)
   cd zepto-ai-capstone
