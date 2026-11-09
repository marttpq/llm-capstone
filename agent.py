import sys
import json
import types
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI
from rag import vector_search
from memory import memory_manager
from tools import calculate_tdee, calculate_1rm

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

# 1. Complete Tool Definitions (OpenAI Function Schemas)
AGENT_TOOLS = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tdee",
            "description": "คำนวณอัตราการเผาผลาญพื้นฐาน (BMR) และพลังงานที่ใช้ต่อวัน (TDEE) พร้อมสัดส่วนสารอาหาร (Macros: โปรตีน คาร์โบไฮเดรต ไขมัน) สำหรับลดไขมันหรือเพิ่มกล้ามเนื้อ",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "น้ำหนักตัวเป็นกิโลกรัม"},
                    "height_cm": {"type": "number", "description": "ส่วนสูงเป็นเซนติเมตร"},
                    "age": {"type": "integer", "description": "อายุเป็นปีบริบูรณ์"},
                    "gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "เพศกำเนิด (male หรือ female)",
                    },
                    "activity_level": {
                        "type": "string",
                        "enum": ["sedentary", "light", "moderate", "active", "very_active"],
                        "description": "ระดับกิจกรรม: sedentary (ไม่ออกกำลังกาย), light (1-3 วัน/สัปดาห์), moderate (3-5 วัน/สัปดาห์), active (6-7 วัน/สัปดาห์), very_active (ออกกำลังกายหนักมาก)",
                    },
                },
                "required": ["weight_kg", "height_cm", "age", "gender"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_1rm",
            "description": "คำนวณน้ำหนักยกได้สูงสุด 1 ครั้ง (One-Rep Max: 1RM) จากน้ำหนักและจำนวนครั้งที่ยกได้จริง พร้อมตารางเปอร์เซ็นต์สำหรับจัดโปรแกรมฝึก",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "น้ำหนักที่ใช้ยก (กก.)"},
                    "reps": {"type": "integer", "description": "จำนวนครั้งที่ยกได้จนหมดแรง"},
                },
                "required": ["weight_kg", "reps"],
            },
        },
    },
]

SYSTEM_PROMPT = """คุณคือผู้ช่วยโค้ชฟิตเนสและโภชนาการ (Fitness Assistant)
กฎสำคัญ:
1. หากผู้ใช้ถามคำถามเกี่ยวกับการออกกำลังกาย ท่าฝึก อาการบาดเจ็บ หรือโภชนาการ คุณต้องเรียกใช้เครื่องมือ `search_fitness_knowledge` เพื่อค้นหาข้อมูลจากคลังความรู้ก่อนตอบเสมอ
2. หากผู้ใช้ต้องการคำนวณแคลอรี่ BMR หรือ TDEE ให้เรียกใช้เครื่องมือ `calculate_tdee` (ต้องการ: น้ำหนัก, ส่วนสูง, อายุ, เพศ)
3. หากผู้ใช้ต้องการคำนวณความแข็งแรงหรือ One-Rep Max ให้เรียกใช้เครื่องมือ `calculate_1rm`
4. เมื่อได้ข้อมูลจากเครื่องมือแล้ว ให้อธิบายคำตอบโดยอ้างอิงข้อมูลและระบุหมายเลขอ้างอิง เช่น [Chunk 1], [Chunk 2] ให้ชัดเจน
5. จดจำและอ้างอิงบทสนทนาก่อนหน้านี้เพื่อให้คำแนะนำได้อย่างต่อเนื่อง
6. หากเป็นการทักทายทั่วไป สามารถตอบทักทายอย่างสุภาพได้โดยไม่ต้องเรียกใช้เครื่องมือ
"""


def execute_search_tool(query: str, client: OpenAI) -> str:
    """Executes vector_search RAG function and returns retrieved chunks with citations."""
    results = vector_search(query=query, client=client, top_k=2)
    if not results:
        return "ไม่พบข้อมูลที่ตรงกับคำค้นหาในคลังความรู้"

    formatted_chunks = []
    for r in results:
        formatted_chunks.append(f"{r['citation']}:\n{r['content']}")
    return "\n\n".join(formatted_chunks)


def execute_tool_call(tool_name: str, args_dict: Dict[str, Any], client: OpenAI) -> str:
    """Dispatches tool execution to the appropriate Python function."""
    if tool_name == "search_fitness_knowledge":
        query = args_dict.get("query", "")
        return execute_search_tool(query=query, client=client)

    elif tool_name == "calculate_tdee":
        result = calculate_tdee(
            weight_kg=float(args_dict.get("weight_kg", 70)),
            height_cm=float(args_dict.get("height_cm", 170)),
            age=int(args_dict.get("age", 25)),
            gender=str(args_dict.get("gender", "male")),
            activity_level=str(args_dict.get("activity_level", "moderate")),
        )
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "calculate_1rm":
        result = calculate_1rm(
            weight_kg=float(args_dict.get("weight_kg", 50)),
            reps=int(args_dict.get("reps", 5)),
        )
        return json.dumps(result, ensure_ascii=False)

    return f"Error: Tool '{tool_name}' not recognized."


def run_fitness_agent(
    user_input: str,
    client: OpenAI,
    session_id: Optional[str] = None,
    max_steps: int = 5,
) -> str:
    """
    ReAct-style Agent Loop with Function Calling and Multi-turn Memory:
    - Restores prior message history for session_id if provided.
    - Interacts with gemma-4-E4B-it at temperature=0.
    - Dynamically calls tools (search_fitness_knowledge, calculate_tdee, calculate_1rm).
    - Terminates when the model returns a direct message or when max_steps is reached.
    """
    # 1. Prepare message history
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Load past session conversation if available
    if session_id:
        past_history = memory_manager.get_history(session_id)
        messages.extend(past_history)

    # Append current user prompt
    messages.append({"role": "user", "content": user_input})

    steps = 0
    while steps < max_steps:
        steps += 1

        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        # Stop Condition: Model returned direct text answer
        if not message.tool_calls:
            final_reply = message.content or ""

            # Save turn to session memory
            if session_id:
                memory_manager.append_message(session_id, {"role": "user", "content": user_input})
                memory_manager.append_message(session_id, {"role": "assistant", "content": final_reply})

            return final_reply

        # Model requested tool calls: append assistant message with tool calls to history
        messages.append(message)

        # Execute each requested tool call
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception:
                args = {}

            tool_result = execute_tool_call(tool_name=func_name, args_dict=args, client=client)

            # Append tool result to message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    return "Error: Agent reached maximum steps limit without completing."
