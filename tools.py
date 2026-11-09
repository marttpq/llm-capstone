import json
from typing import Any, Dict, List
from openai import OpenAI
from rag import vector_search


# ==========================================
# 1. Deterministic Python Tool Functions
# ==========================================

def calculate_tdee(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Calculates BMR and TDEE using the Mifflin-St Jeor formula,
    plus recommended macro breakdowns based on fitness best practices.
    """
    # 1. Base BMR
    if gender.lower() in ["male", "ชาย", "m"]:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # 2. Activity Multiplier
    multipliers = {
        "sedentary": 1.2,       # Little or no exercise
        "light": 1.375,         # Light exercise 1-3 days/week
        "moderate": 1.55,       # Moderate exercise 3-5 days/week
        "active": 1.725,        # Heavy exercise 6-7 days/week
        "very_active": 1.9,     # Very heavy training / physical job
    }
    multiplier = multipliers.get(activity_level.lower(), 1.55)
    tdee = round(bmr * multiplier)

    # 3. Macro Guidelines (Protein: 2.0g/kg based on domain knowledge, Fat: 25%, Carbs: remainder)
    protein_grams = round(weight_kg * 2.0)
    protein_cals = protein_grams * 4
    fat_cals = round(tdee * 0.25)
    fat_grams = round(fat_cals / 9)
    carb_cals = max(0, tdee - protein_cals - fat_cals)
    carb_grams = round(carb_cals / 4)

    return {
        "bmr_kcal": round(bmr),
        "tdee_kcal": tdee,
        "activity_level": activity_level,
        "recommended_macros": {
            "protein_g": protein_grams,
            "fat_g": fat_grams,
            "carbs_g": carb_grams,
        },
        "target_goals": {
            "maintenance_kcal": tdee,
            "fat_loss_deficit_kcal": round(tdee - 400),
            "lean_bulk_surplus_kcal": round(tdee + 300),
        },
    }


def calculate_1rm(weight_kg: float, reps: int) -> Dict[str, Any]:
    """
    Calculates One-Rep Maximum (1RM) using the standard Epley formula:
    1RM = Weight * (1 + Reps / 30)
    """
    if reps <= 0 or weight_kg <= 0:
        return {"error": "น้ำหนักและจำนวนครั้งต้องมากกว่า 0"}

    if reps == 1:
        one_rm = weight_kg
    else:
        one_rm = weight_kg * (1 + reps / 30.0)

    one_rm = round(one_rm, 1)

    return {
        "estimated_1rm_kg": one_rm,
        "tested_weight_kg": weight_kg,
        "tested_reps": reps,
        "training_percentages": {
            "90% (Strength 3-4 reps)": round(one_rm * 0.90, 1),
            "80% (Hypertrophy 7-8 reps)": round(one_rm * 0.80, 1),
            "70% (Endurance 10-12 reps)": round(one_rm * 0.70, 1),
            "60% (Warmup/Deload 15+ reps)": round(one_rm * 0.60, 1),
        },
    }


def search_fitness_kb(query: str, client: OpenAI) -> Dict[str, Any]:
    """
    Searches the in-memory RAG knowledge base for injury guidance,
    training principles, and nutrition cues.
    """
    results = vector_search(query=query, client=client, top_k=2)
    return {
        "query": query,
        "found_chunks": [
            {"citation": r["citation"], "content": r["content"], "score": r["score"]}
            for r in results
        ],
    }


# ==========================================
# 2. OpenAI Function Tool Schemas
# ==========================================

FITNESS_TOOLS = [
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
                        "description": "ระดับกิจกรรมทางกาย: sedentary (ไม่ออกกำลังกาย), light (1-3 วัน/สัปดาห์), moderate (3-5 วัน/สัปดาห์), active (6-7 วัน/สัปดาห์), very_active (ออกกำลังกายหนักมาก/งานใช้แรง)",
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
                    "reps": {"type": "integer", "description": "จำนวนครั้งที่ยกได้จนหมดแรง (Reps to failure)"},
                },
                "required": ["weight_kg", "reps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fitness_kb",
            "description": "ค้นหาข้อมูลในคลังความรู้ฟิตเนสเกี่ยวกับการปรับท่าเมื่อเจ็บข้อมือ เจ็บข้อศอก หลักการ Progressive Overload และแหล่งโปรตีนจากธรรมชาติ",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "คำค้นหา เช่น 'เจ็บข้อมือ', 'progressive overload', 'อกไก่และไข่'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ==========================================
# 3. Tool Dispatcher
# ==========================================

def execute_tool_call(tool_name: str, arguments_json: str, client: OpenAI) -> str:
    """Parses JSON arguments and executes the corresponding Python tool."""
    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
    except Exception as e:
        return json.dumps({"error": f"Invalid JSON arguments: {str(e)}"}, ensure_ascii=False)

    if tool_name == "calculate_tdee":
        result = calculate_tdee(
            weight_kg=float(args["weight_kg"]),
            height_cm=float(args["height_cm"]),
            age=int(args["age"]),
            gender=str(args["gender"]),
            activity_level=str(args.get("activity_level", "moderate")),
        )
    elif tool_name == "calculate_1rm":
        result = calculate_1rm(
            weight_kg=float(args["weight_kg"]),
            reps=int(args["reps"]),
        )
    elif tool_name == "search_fitness_kb":
        result = search_fitness_kb(
            query=str(args["query"]),
            client=client,
        )
    else:
        result = {"error": f"Unknown tool name: {tool_name}"}

    return json.dumps(result, ensure_ascii=False)

