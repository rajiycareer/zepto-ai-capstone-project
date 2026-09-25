# Zepto Customer Support GenAI Service

A lightweight, local GenAI support workflow built for Zepto's delivery, return, and membership policies.

## RAG Pipeline Architecture

1. **Ingestion**: Document texts from `docs/doc_01.txt` through `docs/doc_08.txt` are parsed during startup or build via `ingestion.py`.
2. **Embedding**: Documents are converted to dense vector representations locally using `sentence-transformers/all-MiniLM-L6-v2`. Vectors are indexed in a local `ChromaDB` collection (`zepto_policies`).
3. **Retrieval**: When a query is classified as `policy_question`, `retrieve_and_answer` queries ChromaDB for top-3 relevant chunks using cosine distance. Retrieval always runs for real in both mock and production modes.
4. **Generation**:
   - **Mock Mode (`MOCK_LLM=1` / Unset)**: The pipeline performs deterministic routing and string formatting without external network calls.
     - `classify_intent` uses a keyword heuristic.
     - `retrieve_and_answer` formats the top retrieved text chunk snippet.
     - `direct_answer` returns a fixed canned standard message.
   - **Real LLM Mode (`MOCK_LLM=0`)**:
     - `classify_intent` uses the LLM to classify questions.
     - `retrieve_and_answer` formats the context via `PROMPT_TEMPLATE` and calls the LLM with schema validation and 2 retry attempts on failure.

---

## Local Setup & Execution

### Run Locally with Uvicorn

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860