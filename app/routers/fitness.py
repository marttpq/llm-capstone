from fastapi import APIRouter, HTTPException
from app.schemas.fitness import (
    ChatRequest,
    ChatResponse,
    ExerciseRecommendationRequest,
)
from app.services.llm_service import llm_service
from app.core.config import settings

router = APIRouter(prefix="/fitness", tags=["Fitness Assistant"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):
    """
    General fitness Q&A endpoint powered by the configured LLM.
    """
    try:
        reply = llm_service.generate_chat_reply(request.message)
        return ChatResponse(reply=reply, model=settings.LLM_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-exercise", response_model=ChatResponse)
def recommend_exercise(request: ExerciseRecommendationRequest):
    """
    Get customized exercise suggestions considering muscle group, equipment,
    fitness level, and injury limitations.
    """
    try:
        reply = llm_service.generate_exercise_recommendation(
            muscle_group=request.muscle_group,
            fitness_level=request.fitness_level or "beginner",
            injuries=request.injuries_or_limitations,
            equipment=request.equipment,
        )
        return ChatResponse(reply=reply, model=settings.LLM_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

