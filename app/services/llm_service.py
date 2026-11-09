from openai import OpenAI
from app.core.config import settings
from app.core.prompts import FITNESS_ASSISTANT_SYSTEM_PROMPT


class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
        )
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE

    def generate_chat_reply(self, user_message: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FITNESS_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""

    def generate_exercise_recommendation(
        self,
        muscle_group: str,
        fitness_level: str,
        injuries: str | None,
        equipment: str | None,
    ) -> str:
        prompt_parts = [
            f"ช่วยแนะนำท่าออกกำลังกาย/เวทเทรนนิ่งสำหรับกล้ามเนื้อ: {muscle_group}",
            f"ระดับความชำนาญ: {fitness_level}",
        ]
        if equipment:
            prompt_parts.append(f"อุปกรณ์ที่มี: {equipment}")
        if injuries:
            prompt_parts.append(
                f"ข้อจำกัดหรืออาการบาดเจ็บ: {injuries} (โปรดเลือกท่าที่ปลอดภัย ไม่สร้างแรงกดทับจุดนี้โดยตรง)"
            )
        prompt_parts.append(
            "ขอคำแนะนำอย่างละเอียด พร้อมวิธีจัดท่า (Setup), วิธีฝึก (Execution), และข้อควรระวัง (Cues)"
        )

        user_content = "\n".join(prompt_parts)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FITNESS_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""


llm_service = LLMService()

