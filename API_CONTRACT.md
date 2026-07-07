# Oracle Finance — REST API Contract (v1)

Locked at Checkpoint 1 so frontend and backend can develop in parallel.
Base URL: `http://localhost:8000` · All non-health routes under `/api/v1`.

## Envelope

Every endpoint returns:
```json
{ "success": true, "data": { ... }, "error": null }
```
On failure: `{ "success": false, "data": null, "error": {"code": "...", "message": "..."} }`

## Endpoints

### Health
- `GET /health` → `{status, app}`
- `GET /health/components` → `{mongo, chroma, llm_configured, llm_model}`

### Chat (Checkpoint 3)
`POST /api/v1/chat`
```json
// request
{ "query": "Would Buffett buy ITC at today's P/E?", "persona": "buffett" | "auto", "session_id": "optional" }
// response.data
{
  "answer": "...",
  "persona": "buffett",
  "citations": [
    { "chunk_id": "brk1989-014", "book": "Berkshire Letter 1989", "author": "Warren Buffett", "page": 4, "passage": "..." }
  ],
  "contexts": ["raw retrieved chunks..."],
  "faithfulness": 0.92        // null until Checkpoint 6
}
```

### Personas (Checkpoint 4)
- `GET /api/v1/personas` → `[{id, name, philosophy, source_books[]}]`

### Market (Checkpoint 5)
- `GET /api/v1/market/quote/{symbol}` → `{symbol, price, pe, market_cap, currency}`
- `GET /api/v1/market/macro` → `{indicators: {...}}`

### Debate (Checkpoint 7)
`POST /api/v1/debate`
```json
// request
{ "topic": "Buy RELIANCE.NS at current valuation", "persona_a": "buffett", "persona_b": "ackman", "rounds": 2 }
// response.data
{ "transcript": [ {"persona": "buffett", "argument": "...", "citations": [...]}, ... ] }
```

### Portfolio (Checkpoint 7)
- `GET/POST/DELETE /api/v1/portfolio/holdings`
- `GET /api/v1/portfolio/thesis-check` → per-holding, per-persona verdicts

### Ingestion (Checkpoint 7)
- `POST /api/v1/ingest/upload` (multipart PDF) → `{doc_id, chunks_indexed}`

## Citation object (canonical)
```json
{ "chunk_id": "str", "book": "str", "author": "str", "page": 0, "passage": "str", "score": 0.0 }
```

## Chunk metadata contract (ChromaDB)
`{book, author, page, chunk_id, persona_tag}` — set at ingestion (Checkpoint 2), consumed by retrieval + citation renderer.
