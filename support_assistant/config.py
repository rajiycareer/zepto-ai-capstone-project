import os
from pydantic import BaseModel, Field
from typing import List

# Read MOCK_LLM env var (Default: "1")
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float