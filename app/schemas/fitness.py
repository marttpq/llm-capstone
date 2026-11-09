from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User question or prompt for the fitness assistant",
        examples=["ขอท่าเวทเทรนนิ่ง 1 ท่าสำหรับคนเจ็บข้อมือ"],
    )


class ChatResponse(BaseModel):
    reply: str
    model: str


class ExerciseRecommendationRequest(BaseModel):
    muscle_group: str = Field(
        ...,
        description="Target muscle group (e.g. ขา, อก, หลัง, ไหล่, แขน)",
        examples=["อก"],
    )
    fitness_level: Optional[str] = Field(
        default="beginner",
        description="Fitness level (beginner, intermediate, advanced)",
        examples=["beginner"],
    )
    injuries_or_limitations: Optional[str] = Field(
        default=None,
        description="Injuries or physical limitations",
        examples=["เจ็บข้อมือ"],
    )
    equipment: Optional[str] = Field(
        default="bodyweight",
        description="Available equipment (e.g. bodyweight, dumbbell, machine)",
        examples=["dumbbell"],
    )

