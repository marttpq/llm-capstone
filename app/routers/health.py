from fastapi import APIRouter, HTTPException
from app.services.llm_service import llm_service
from app.core.config import settings

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Fitness Assistant API",
        "model": settings.LLM_MODEL,
    }


@router.get("/test-llm")
def test_llm():
    try:
        reply = llm_service.generate_chat_reply("ขอท่าเวทเทรนนิ่ง 1 ท่าสำหรับคนเจ็บข้อมือ")
        return {
            "model": settings.LLM_MODEL,
            "prompt": "ขอท่าเวทเทรนนิ่ง 1 ท่าสำหรับคนเจ็บข้อมือ",
            "response": reply,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

