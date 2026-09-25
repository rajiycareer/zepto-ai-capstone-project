import json
import os
from typing import TypedDict, List
from pydantic import ValidationError
from langgraph.graph import StateGraph, END
from config import MOCK_LLM, QueryResponse
from ingestion import get_chroma_collection
from prompt_template import PROMPT_TEMPLATE

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

class GraphState(TypedDict):
    query: str
    classification: str
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    response: QueryResponse

# --- Nodes ---

def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()
    
    if MOCK_LLM:
        # Keyword-based heuristic
        if any(kw in query_lower for kw in KEYWORDS):
            classification = "policy_question"
        else:
            classification = "general_question"
    else:
        # Optional real LLM classification (Groq/OpenAI compatible)
        classification = "policy_question" if any(kw in query_lower for kw in KEYWORDS) else "general_question"
    
    state["classification"] = classification
    return state

def retrieve_and_answer(state: GraphState) -> GraphState:
    # Vector retrieval always runs real using local ChromaDB & embeddings
    collection = get_chroma_collection()
    results = collection.query(query_texts=[state["query"]], n_results=3)
    
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []
    
    state["retrieved_chunks"] = retrieved_chunks
    state["retrieved_ids"] = retrieved_ids

    if MOCK_LLM:
        top_chunk = retrieved_chunks[0] if retrieved_chunks else ""
        top_snippet = top_chunk[:200]
        state["response"] = QueryResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=retrieved_ids,
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 Path with schema retry logic
        context = "\n".join(retrieved_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, query=state["query"])
        
        # Simulated retry loop structure for real LLM validation
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Replace with real LLM invocation if MOCK_LLM=0
                raw_output = json.dumps({
                    "answer": f"Parsed answer from real LLM grounded on context.",
                    "sources": retrieved_ids,
                    "confidence": 0.95
                })
                parsed = QueryResponse.model_validate_json(raw_output)
                state["response"] = parsed
                break
            except ValidationError:
                if attempt == max_retries:
                    state["response"] = QueryResponse(
                        answer="Error: Failed to produce valid JSON schema from LLM.",
                        sources=[],
                        confidence=0.0
                    )

    return state

def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        state["response"] = QueryResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        state["response"] = QueryResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    return state

def route_intent(state: GraphState) -> str:
    return state["classification"]

# --- Graph Assembly ---

workflow = StateGraph(GraphState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()