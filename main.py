import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Fitness Assistant API")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fitness Assistant API is running"}


@app.get("/test-llm")
def test_llm():
    try:
        completion = client.chat.completions.create(
            model="gemma-4-E4B-it",
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

