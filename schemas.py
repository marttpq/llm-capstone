from typing import List
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    citation: str = Field(..., description="Citation marker, e.g. [Chunk 1]")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    content: str = Field(..., description="Text content of the retrieved chunk")


class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User question to ask the fitness assistant",
        examples=["เจ็บข้อมือเล่นอกท่าไหนดี"],
    )
    top_k: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of context chunks to retrieve for grounding",
    )


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]

