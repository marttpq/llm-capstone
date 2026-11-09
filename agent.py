import sys
import json
import types
import os
from typing import Any, Dict
from openai import OpenAI
from rag import vector_search

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

# Model configuration
AGENT_MODEL = os.getenv("LLM_MODEL", "gemma-4-E4B-it")

# 1. Tool Definition (OpenAI Schema format)
SEARCH_FITNESS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_fitness_knowledge",
        "description": "ค้นหาข้อมูลในคลังความรู้ฟิตเนสเกี่ยวกับการออกกำลังกาย การปรับท่าเมื่อมีอาการบาดเจ็บ (เช่น ข้อมือ ข้อศอก) หลักการ Progressive Overload และโภชนาการ (เช่น โปรตีน อกไก่ ไข่)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "คำค้นหาสำหรับค้นหาในคลังความรู้ฟิตเนส",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """คุณคือผู้ช่วยโค้ชฟิตเนสและโภชนาการ (Fitness Assistant)
กฎสำคัญ:
1. หากผู้ใช้ถามคำถามเกี่ยวกับการออกกำลังกาย ท่าฝึก อาการบาดเจ็บ หรือโภชนาการ คุณต้องเรียกใช้เครื่องมือ `search_fitness_knowledge` เพื่อค้นหาข้อมูลจากคลังความรู้ก่อนตอบเสมอ
2. เมื่อได้ข้อมูลจากเครื่องมือแล้ว ให้อธิบายคำตอบโดยอ้างอิงข้อมูลจากคลังความรู้ และระบุหมายเลขอ้างอิง เช่น [Chunk 1], [Chunk 2] ให้ชัดเจน
3. หากคำถามเป็นการทักทายทั่วไป สามารถตอบทักทายอย่างสุภาพได้โดยไม่ต้องเรียกใช้เครื่องมือ
"""


def execute_search_tool(query: str, client: OpenAI) -> str:
    """
    Executes the vector_search RAG function and returns retrieved chunks with citations.
    Uses bge-m3 as configured in rag.py.
    """
    results = vector_search(query=query, client=client, top_k=2)
    if not results:
        return "ไม่พบข้อมูลที่ตรงกับคำค้นหาในคลังความรู้"

    formatted_chunks = []
    for r in results:
        formatted_chunks.append(f"{r['citation']}:\n{r['content']}")
    return "\n\n".join(formatted_chunks)


def run_fitness_agent(user_input: str, client: OpenAI, max_steps: int = 5) -> str:
    """
    ReAct-style Agent Loop with Function Calling:
    - Interacts with gemma-4-E4B-it at temperature=0.
    - Uses search_fitness_knowledge tool when needed.
    - Terminates when the model returns a direct message or when max_steps is reached.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    steps = 0
    while steps < max_steps:
        steps += 1

        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=[SEARCH_FITNESS_TOOL],
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        # Stop Condition: If the model returns a direct message without tool calls
        if not message.tool_calls:
            return message.content or ""

        # Model requested tool calls: append assistant message with tool calls to history
        messages.append(message)

        # Execute each requested tool call
        for tool_call in message.tool_calls:
            if tool_call.function.name == "search_fitness_knowledge":
                try:
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", user_input)
                except Exception:
                    query = user_input

                # Run RAG vector search
                tool_result = execute_search_tool(query=query, client=client)

                # Append tool result to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

    return "Error: Agent reached maximum steps limit without completing."
