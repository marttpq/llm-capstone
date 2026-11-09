import sys
import json
import types
import uuid

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
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from rag import init_knowledge_base, vector_search
from agent import run_fitness_agent
from memory import memory_manager

load_dotenv()

app = FastAPI(title="Fitness Assistant API - ReAct Agent with Function Calling & Memory")

# Initialize OpenAI client with environment settings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User question or prompt for the fitness agent",
        examples=["เจ็บข้อมือเล่นอกท่าไหนดี"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversations",
        examples=["session-123"],
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.on_event("startup")
def startup_event():
    """
    Startup event: Load knowledge.txt and compute chunk embeddings (bge-m3) into memory once.
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


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Returns the frontend chat interface."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/search")
def search(
    query: str = Query(..., description="The query to search in the knowledge base"),
    top_k: int = Query(default=2, ge=1, le=10, description="Number of chunks to return"),
):
    """
    Vector search endpoint from Phase 2:
    Returns the top-k relevant chunks with citations.
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


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Phase 3: ReAct Agent Endpoint with Function Calling & Multi-turn Session Memory.
    Dynamically executes tools (RAG search, TDEE calculation, 1RM estimation)
    and remembers conversation history across turns.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Assign or keep session_id
    active_session_id = request.session_id or str(uuid.uuid4())

    try:
        final_answer = run_fitness_agent(
            user_input=request.message,
            client=client,
            session_id=active_session_id,
        )
        return ChatResponse(response=final_answer, session_id=active_session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/session/{session_id}")
def clear_session(session_id: str):
    """Clears conversation history for a given session."""
    memory_manager.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
