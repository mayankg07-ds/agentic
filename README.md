# Oracle Finance

A Multi-Investor RAG Agent for grounded financial advice, portfolio guidance, and live market analysis.
Every answer is retrieved from the writings of legendary investors (Buffett, Lynch, Dalio, Graham, Ackman) and **cited to the exact passage, book, and page**.

> Plan & checklists: [PROJECT_PLAN.md](PROJECT_PLAN.md) · API spec: [API_CONTRACT.md](API_CONTRACT.md)

## Stack
- **Backend:** FastAPI (Python), MongoDB (Motor), ChromaDB
- **LLM:** NVIDIA NIM — Nemotron (OpenAI-compatible API)
- **Embeddings:** sentence-transformers (local, free)
- **Frontend:** React (Vite) + TailwindCSS + Axios
- **Eval:** RAGAS faithfulness scoring

## Quick start

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env           # then fill in NVIDIA_API_KEY etc.
uvicorn app.main:app --reload --port 8000
```
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health · Components: http://localhost:8000/health/components

### Frontend
```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## Repo layout
```
backend/     FastAPI app (app/core = config, llm, db, vectorstore; app/routers = endpoints)
frontend/    React chat UI + citation renderer
ingestion/   PDF → chunks → ChromaDB pipeline (Checkpoint 2)
eval/        RAGAS eval set + benchmark scripts (Checkpoint 6)
notebooks/   experiments
```
