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

import math
import os
from typing import Any, Dict, List
from openai import OpenAI

# In-memory vector store
DOC_CHUNKS: List[str] = []
DOC_EMBEDDINGS: List[List[float]] = []

# Model defaults
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "bge-m3")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-E4B-it")


def load_and_chunk_data(filepath: str) -> List[str]:
    """
    Reads a domain-specific text file and splits it into paragraph chunks.
    Filters out empty lines.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline to separate paragraphs cleanly
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks


def get_embedding(
    text: str, client: OpenAI, model: str = DEFAULT_EMBEDDING_MODEL
) -> List[float]:
    """
    Generates an embedding vector for a single string using OpenAI client.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_embeddings_batch(
    texts: List[str], client: OpenAI, model: str = DEFAULT_EMBEDDING_MODEL
) -> List[List[float]]:
    """
    Generates embeddings for a list of texts in a single batch API call.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates cosine similarity between two numeric vectors using pure Python/math.
    cos_sim = (A . B) / (||A|| * ||B||)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def init_knowledge_base(
    filepath: str, client: OpenAI, model: str = DEFAULT_EMBEDDING_MODEL
) -> None:
    """
    Loads text chunks from file, computes their embeddings, and stores them in memory.
    Called once during application startup.
    """
    global DOC_CHUNKS, DOC_EMBEDDINGS
    DOC_CHUNKS = load_and_chunk_data(filepath)
    if DOC_CHUNKS:
        DOC_EMBEDDINGS = get_embeddings_batch(DOC_CHUNKS, client, model=model)
    else:
        DOC_EMBEDDINGS = []


def vector_search(
    query: str,
    client: OpenAI,
    top_k: int = 2,
    chunks: List[str] | None = None,
    chunk_embeddings: List[List[float]] | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    """
    Performs vector search:
    1. Embeds the query string.
    2. Computes cosine similarity against document chunk embeddings.
    3. Returns top_k most similar chunks with citation strings.
    """
    target_chunks = chunks if chunks is not None else DOC_CHUNKS
    target_embeddings = (
        chunk_embeddings if chunk_embeddings is not None else DOC_EMBEDDINGS
    )

    if not target_chunks or not target_embeddings:
        return []

    # 1. Embed query
    query_vector = get_embedding(query, client, model=model)

    # 2. Compute similarity for each chunk
    scored_results = []
    for idx, (chunk, emb) in enumerate(zip(target_chunks, target_embeddings)):
        similarity = cosine_similarity(query_vector, emb)
        scored_results.append({
            "citation": f"[Chunk {idx + 1}]",
            "score": round(similarity, 4),
            "content": chunk,
        })

    # Sort descending by similarity score
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # 3. Return top_k
    return scored_results[:top_k]


def generate_rag_response(
    query: str,
    client: OpenAI,
    top_k: int = 2,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline:
    1. Retrieves relevant context chunks via vector search.
    2. Injects context and citation instructions into the LLM prompt.
    3. Returns the grounded answer with source citations.
    """
    # 1. Retrieval
    sources = vector_search(query=query, client=client, top_k=top_k)

    # 2. Construct Augmented Context Block
    context_blocks = []
    for s in sources:
        context_blocks.append(f"{s['citation']}:\n{s['content']}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือ 'FitCoach AI' ผู้ช่วยโค้ชฟิตเนสและเวทเทรนนิ่งมืออาชีพ\n"
        "คำสั่งในการตอบคำถาม:\n"
        "1. ตอบคำถามของผู้ใช้โดยอิงข้อมูลจาก 'บริบทความรู้ (Context)' ที่ให้มาเป็นหลัก\n"
        "2. ต้องระบุเครื่องหมายอ้างอิงท้ายข้อความหรือหัวข้อ เช่น [Chunk 1], [Chunk 2] ให้สอดคล้องกับแหล่งที่มา\n"
        "3. หากข้อมูลในบริบทไม่ครอบคลุม ให้ตอบเท่าที่มีในบริบทอย่างปลอดภัย และระบุข้อควรระวังสำหรับผู้ฝึก\n"
        "4. สื่อสารชัดเจน เข้าใจง่าย และจัดลำดับเป็นข้อๆ"
    )

    user_prompt = f"บริบทความรู้ (Context):\n{context_text}\n\nคำถามของผู้ใช้:\n{query}"

    # 3. Generation
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    answer = completion.choices[0].message.content or ""

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }
