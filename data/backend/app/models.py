from pydantic import BaseModel,ConfigDict
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] =None  # optional now, fallback exists

class QueryResponse(BaseModel):
    answer: str
   

    sources: Optional[List[str]] = None

class IngestResponse(BaseModel):
    message: str
    num_docs: int
    num_chunks: Optional[int] = None

class IngestTextRequest(BaseModel):
    filename: str
    content: str