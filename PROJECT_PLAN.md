# Oracle Finance — Build Plan (Checkpoints & Checklists)

> A Multi-Investor RAG Agent for grounded financial advice, portfolio guidance, and live market analysis.
> Built AI-assisted. Ordered by **dependency**, not calendar. Finish a checkpoint before starting the next.

---

## 0. Scope Tiers (decide what you're actually building)

Don't build everything at once. Tag each feature and protect the core.

- **CORE (must ship):** ingestion → vector DB, RAG Q&A with **citations**, 3 personas, live
  market grounding, chat UI, RAGAS faithfulness score. This alone is a complete project.
- **PLUS (build after core is solid):** Debate Mode, Portfolio Thesis Tracker, PDF upload
  pipeline, all 5 personas.
- **STRETCH (only if core+plus done):** LoRA fine-tuned Mistral-7B router, Historical
  Backtester, Indian-market cross-validation, research paper.

> Biggest de-risk: use a **prompt/embedding-similarity persona classifier** for routing (CORE),
> and treat **LoRA fine-tuning** as STRETCH only. LoRA is the single most likely thing to sink the timeline.

---

## Checkpoint 1 — Foundations & scaffolding
**Exit criterion:** repo runs; FastAPI + MongoDB + ChromaDB all reachable with a "hello world".

- [ ] `git init`, monorepo: `/backend` `/frontend` `/ingestion` `/eval` `/notebooks`
- [ ] FastAPI skeleton with `/health`, Pydantic models, CORS
- [ ] MongoDB Atlas M0 free cluster connected (Motor async driver)
- [ ] ChromaDB installed and persisting locally
- [ ] `.env` + config loader for API keys (Gemini, FRED, Mongo URI)
- [ ] `LLMClient` abstraction (swap Gemini/Claude behind one interface) — do NOT hard-wire one LLM
- [ ] **Team: lock the REST API contract** (endpoints + request/response JSON) so front/back move in parallel
- [ ] React (Vite) + Tailwind app boots, Axios wired to backend

## Checkpoint 2 — Corpus & ingestion pipeline
**Exit criterion:** a raw similarity query returns sensible passages (LLM not involved yet).

- [ ] Collect **free/legal corpus first**: all Berkshire letters (1977–2023), Dalio *Principles* PDF
- [ ] Copyrighted books (Intelligent Investor, One Up on Wall Street, Security Analysis, etc.)
      kept **local/demo only** — noted for the report, not redistributed
- [ ] PyMuPDF parser extracting text **+ page numbers**
- [ ] Chunking (recursive/semantic, ~500–1000 tokens, 10–15% overlap)
- [ ] Rich metadata per chunk: `book`, `author`, `page`, `chunk_id`, `persona_tag`
- [ ] Embeddings via local `sentence-transformers` (free) or `text-embedding-004`
- [ ] Index into ChromaDB; verify with a manual query script
- [ ] Target **10+ sources** indexed, weighted toward free-and-legal ones

## Checkpoint 3 — RAG Q&A with citations (the headline feature)
**Exit criterion:** ask a question → get an answer with exact passage + book + author + page, verified against retrieved chunks.

- [ ] Retrieval endpoint: query → top-k chunks
- [ ] (Optional) reranking step for quality
- [ ] Prompt template forcing **structured citations tied to retrieved `chunk_id`s**
- [ ] Post-generation guard: verify each cited chunk was actually in the retrieved set (no fabricated cites)
- [ ] `/chat` endpoint returns `{answer, citations[], contexts[]}`
- [ ] React chat UI + **Citation Renderer** component (click to expand source passage)
- [ ] One persona (Buffett) working end-to-end — **this is the core demo milestone**

## Checkpoint 4 — Investor personas & routing
**Exit criterion:** user picks or is auto-routed to a persona; each gives distinctly different, in-character, sourced answers.

- [ ] Persona definitions: Buffett (value/moat), Lynch (PEG/growth), Dalio (macro/risk parity),
      Graham (deep value), Ackman (activist) — distinct system prompts + retrieval filters
- [ ] Manual persona switcher in UI
- [ ] Auto persona router = **prompt/embedding classifier** (CORE) — not LoRA
- [ ] Each persona retrieves preferentially from its own author's texts (metadata filter)

## Checkpoint 5 — Live market grounding
**Exit criterion:** "Would Buffett buy this at today's P/E?" pulls real numbers and reasons over them.

- [ ] yfinance integration: price, P/E, market cap, fundamentals
- [ ] FRED API for macro indicators (rates, inflation) — needs free API key
- [ ] NSEpy for Indian NSE/BSE data
- [ ] Grounding layer injects live metrics into the persona prompt
- [ ] Caching for market data (avoid rate limits / repeat calls)

## Checkpoint 6 — RAGAS hallucination guard
**Exit criterion:** every answer shows a faithfulness score; you have a benchmark table.

- [ ] Build hand-labeled eval set (~30–50 Q&A pairs)
- [ ] RAGAS wired: faithfulness + answer-relevancy (no ground truth needed),
      context-precision/recall (needs ground truth)
- [ ] Faithfulness score **displayed as a badge on every chat response**
- [ ] Benchmark report: baseline LLM vs RAG (vs fine-tuned router if STRETCH done)

## Checkpoint 7 — PLUS features
**Exit criterion:** core is rock-solid; these add demo wow-factor.

- [ ] **Debate Mode:** two persona agents argue opposite sides, each sourced only from its own texts
- [ ] **Portfolio Thesis Tracker:** user adds holdings → each evaluated through each persona's
      framework → alert when market conditions break the thesis
- [ ] **Document Ingestion UI:** user uploads a PDF → auto chunk + embed + queryable in same session

## Checkpoint 8 — STRETCH / research contributions
**Exit criterion:** only start if Checkpoints 1–7 are done and stable.

- [ ] LoRA / QLoRA fine-tune Mistral-7B as persona router (needs 16GB+ GPU) — compare vs classifier
- [ ] Historical Backtester: test one cited rule (e.g. Lynch PEG<1) on NSE/BSE historical data
- [ ] Cross-market thesis validation (US philosophy vs Indian market data)
- [ ] Research paper: multi-persona RAG + RAGAS hallucination evaluation

## Checkpoint 9 — Harden, deploy, document
- [ ] Caching + async perf pass
- [ ] Dockerize backend; deploy backend (Render/Railway), frontend (Vercel/Netlify)
- [ ] Error handling, rate-limit handling, loading states in UI
- [ ] README + architecture diagram + final RAGAS benchmark report
- [ ] Demo script / walkthrough

---

## Resources Required

### Accounts / software (all have free tiers)
| Need | Choice | Cost |
|---|---|---|
| LLM generation | Google Gemini API (free Flash tier) | Free → ~$0–20/mo if heavy |
| LLM fallback | Anthropic Claude API (Haiku cheap) | Pay-as-you-go, optional |
| Embeddings | `sentence-transformers` local / `text-embedding-004` | Free (local) |
| Vector DB | ChromaDB (local) | Free |
| Vector DB (prod, optional) | Pinecone starter | Free tier |
| Database | MongoDB Atlas M0 | Free |
| Market data | yfinance, NSEpy | Free |
| Macro data | FRED API (free key) | Free |
| Eval | RAGAS (OSS) | Free |
| PDF parsing | PyMuPDF | Free |
| Hosting | Render/Railway + Vercel/Netlify + HF Spaces | Free tiers |
| CI / VCS | GitHub + Actions | Free |

**Realistic cash budget: ₹0 for CORE.** Only likely paid items: Colab Pro (~₹1,000/mo) or
free Kaggle GPU for LoRA, plus minor LLM API overage. The whole core can run on ₹0.

### Compute
- **Dev:** any 8–16GB RAM laptop (ChromaDB + local embeddings run on CPU).
- **LoRA fine-tuning (STRETCH):** 16GB+ VRAM GPU. Options, best first:
  1. University ML lab GPU (ask early), 2. Kaggle (free T4×2, 30h/wk),
  3. Colab Pro (paid), 4. rent A10/A100 hourly (RunPod/Lambda) for the final run only.
  Use **QLoRA (4-bit)** to fit Mistral-7B in ~12–16GB.

### Data / copyright (important)
- **Free & publishable:** Berkshire letters 1977–2023, Dalio *Principles* (free PDF),
  SEC EDGAR filings, public investor letters/essays, FRED macro data.
- **Copyrighted — local/demo only, cite in report, don't redistribute:** *Intelligent Investor*,
  *One Up on Wall Street*, *Security Analysis*, *Common Stocks and Uncommon Profits*.
- Get faculty guide sign-off on the data approach before any **public** deployment.

### Skills to line up
- Python async + FastAPI + Pydantic · RAG fundamentals (chunking, embeddings, vector search,
  reranking, prompt design) · React + Tailwind + streaming responses · MongoDB modeling ·
  (STRETCH) HuggingFace PEFT/LoRA + bitsandbytes · Docker + one deploy target.

### Team split (from PDF)
- **A — RAG/LLM:** ingestion, chunking, vector DB, retrieval/rerank, RAGAS, LoRA.
- **B — Agent Logic:** persona prompts/routing, Debate Mode, market grounding, portfolio loop, backtester.
- **C — Frontend/Data/DevOps:** React chat + citation renderer, dashboard, upload UI, Mongo layer, Docker/deploy/CI.

---

## Top Risks
- **LoRA eats the schedule** → keep it STRETCH; ship classifier router as CORE.
- **Copyright on books** → free corpus for public deploy, copyrighted local-only.
- **Weak retrieval** → invest in chunking + reranking early; measure with RAGAS.
- **Citation fabrication** → verify cited chunks were actually retrieved.
- **Scope creep** → enforce tiers; don't start PLUS until CORE demos cleanly.
