from fastapi import FastAPI
from config import QueryRequest, QueryResponse
from graph import app_graph
from ingestion import initialize_vector_store

app = FastAPI(title="Zepto Customer Support GenAI Service")

@app.on_event("startup")
def startup_event():
    initialize_vector_store()

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "classification": "",
        "retrieved_chunks": [],
        "retrieved_ids": [],
        "response": None
    }
    final_state = app_graph.invoke(initial_state)
    return final_state["response"]