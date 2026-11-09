# Fitness LLM Agent

An intelligent, full-stack Fitness Assistant built with **FastAPI**, **In-Memory Retrieval-Augmented Generation (RAG)**, and an autonomous **ReAct Agent Loop with Function Calling**.

---

## 📖 Description

**Fitness LLM Agent** is designed to provide accurate, safe, and context-grounded recommendations for fitness, diet, and joint-safe exercise programming. 

Many general LLMs hallucinate training advice or suggest biomechanically hazardous exercises to individuals with joint pain. This project solves that problem by integrating a verified domain knowledge base covering:
- **Joint-Safe Exercise Adaptations**: Tailored biomechanics and modifications for wrist joint pain (avoiding heavy straight-bar presses, adopting neutral grip and machine chest flyes) and elbow joint/tendonitis issues.
- **Hypertrophy & Progressive Overload**: Safe overload variables (reps, sets, tempo, time under tension) beyond just adding weight.
- **Evidence-Based Nutrition**: Protein requirements for muscle protein synthesis (MPS) (1.6 to 2.2 g/kg/day) and bioavailable whole-food protein sources (chicken breast, eggs).

---

## ⚡ Features

- **In-Memory RAG (`bge-m3`)**:
  - High-precision dense semantic retrieval pre-indexed into memory at application startup.
  - Zero heavy external database dependencies: implements pure Python vector math for cosine similarity.
- **Autonomous Agent Tool Calling (`gemma-4-E4B-it`)**:
  - ReAct-style agent loop utilizing OpenAI-compatible Function Calling.
  - Dynamically triggers the `search_fitness_knowledge` tool whenever a user asks about exercises, injuries, or nutrition, citing verified source chunks (`[Chunk 1]`, `[Chunk 2]`).
  - Multi-turn conversational memory supported across turns.
- **Reproducible QA Evaluation Harness**:
  - 30-case benchmark dataset (`eval_set.json`) testing keyword recall.
  - Automated evaluation comparing Baseline (`top_k=1`, 93.3%) against Improved (`top_k=3`, 100.0%).
- **Interactive Web UI**:
  - Clean, responsive single-page chat interface (`index.html`) served directly from the FastAPI root endpoint.

---

## 🚀 How to Run

Follow these step-by-step instructions to get the application running locally:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root directory (refer to `.env.example`):
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://llm.nat-d.uk/v1
LLM_MODEL=gemma-4-E4B-it
OPENAI_EMBEDDING_MODEL=bge-m3
```

### 3. Start the Server
Run the FastAPI development server with Uvicorn:
```bash
python -m uvicorn main:app --reload
```
At startup, the server automatically reads `knowledge.txt` and embeds all chunks into memory using `bge-m3`.

### 4. Access the UI
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
You can now chat directly with the Fitness Agent through the web interface!

---

## 📊 How to Run Evaluation

We provide a standalone evaluation script to verify retrieval hit rate and recall across 30 domain-specific benchmark questions:

```bash
python evaluate.py
```

### Benchmark Results Summary

| Configuration | Hits / Total | Hit Rate (%) | Avg Latency |
| :--- | :---: | :---: | :---: |
| **Baseline (`top_k=1`)** | 28 / 30 | 93.33% | ~242 ms |
| **Improved (`top_k=3`)** | **30 / 30** | **100.00%** | ~269 ms |
| **Delta Improvement** | **+2 hits** | **+6.67% absolute** | **+7.14% relative** |

> **Evaluation Insight:** Expanding context from `top_k=1` to `top_k=3` eliminated false-negative boundary misses between closely related upper-body joint queries, achieving a perfect **100% Hit Rate**.

---

## 🧪 Automated Integration Tests

Run the complete test suite verifying health checks, vector search, validation errors, and the agent loop:

```bash
python -m unittest test_api.py
```

---

## 📁 Project Structure

```text
.
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore specification
├── Dockerfile            # Container deployment definition
├── README.md             # Project documentation & presentation
├── agent.py              # ReAct Agent loop with Function Calling
├── eval_set.json         # 30-case benchmark evaluation dataset
├── evaluate.py           # Automated evaluation harness
├── index.html            # Web chat UI (Tailwind CSS)
├── knowledge.txt         # Domain knowledge base
├── main.py               # FastAPI application entrypoint & routers
├── memory.py             # Multi-turn session memory manager
├── rag.py                # In-memory vector search & cosine similarity
├── requirements.txt      # Project dependencies
├── schemas.py            # Pydantic request & response schemas
├── test_api.py           # Automated integration test suite
└── tools.py              # Tool schemas & fitness calculation functions
```
