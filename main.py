import sys
import json
import types

# Windows WDAC / AppLocker fallback for jiter binary DLL
try:
    import jiter  # noqa: F401
except ImportError:
    jiter_shim = types.ModuleType("jiter")
    jiter_shim.from_json = lambda b, **kwargs: json.loads(
        b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b
    )
    jiter_shim.cache_clear = lambda: None
    sys.modules["jiter"] = jiter_shim

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from rag import init_knowledge_base, vector_search, generate_rag_response
from schemas import RAGQueryRequest, RAGQueryResponse

load_dotenv()

app = FastAPI(title="Fitness Assistant API with Lightweight RAG")

# Initialize OpenAI client with environment settings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)


@app.on_event("startup")
def startup_event():
    """
    Startup event: Load knowledge.txt and compute chunk embeddings into memory once.
    """
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
    if os.path.exists(knowledge_path):
        try:
            print(f"Indexing '{knowledge_path}' into in-memory vector store...")
            init_knowledge_base(knowledge_path, client)
            print("Knowledge base loaded and indexed successfully.")
        except Exception as e:
            print(f"Warning: Failed to index knowledge base on startup: {e}")
    else:
        print(f"Warning: Knowledge file not found at {knowledge_path}")


@app.get("/")
def read_root():
    """Health check / root status."""
    return {"status": "ok", "message": "Fitness Assistant API is running"}


@app.get("/test-llm")
def test_llm():
    """Test endpoint verifying standard LLM generation without RAG."""
    try:
        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gemma-4-E4B-it"),
            messages=[
                {
                    "role": "user",
                    "content": "ขอท่าเวทเทรนนิ่ง 1 ท่าสำหรับคนเจ็บข้อมือ",
                }
            ],
            temperature=0,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search(
    query: str = Query(..., description="The query to search in the knowledge base"),
    top_k: int = Query(default=2, ge=1, le=10, description="Number of chunks to return"),
):
    """
    Vector search endpoint:
    Embeds the user query, calculates cosine similarity against in-memory chunk
    embeddings, and returns the top-k relevant chunks with citations.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = vector_search(query=query, client=client, top_k=top_k)
        return {
            "query": query,
            "top_k": top_k,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    """
    End-to-end RAG endpoint:
    Retrieves relevant knowledge chunks, injects them into the LLM prompt,
    and returns an answer grounded in domain knowledge with source citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = generate_rag_response(
            query=request.query,
            client=client,
            top_k=request.top_k,
        )
        return RAGQueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
