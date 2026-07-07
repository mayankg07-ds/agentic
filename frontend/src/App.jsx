import { useEffect, useState } from 'react'
import { getComponentHealth } from './services/api'

function StatusPill({ label, ok }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800/60 px-4 py-3">
      <span
        className={`h-2.5 w-2.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-rose-400'}`}
      />
      <span className="text-sm text-slate-200">{label}</span>
      <span className={`ml-auto text-xs ${ok ? 'text-emerald-400' : 'text-rose-400'}`}>
        {ok ? 'up' : 'down'}
      </span>
    </div>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getComponentHealth()
      .then(setHealth)
      .catch((e) => setError(e.message))
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="text-3xl font-bold tracking-tight">
          Oracle <span className="text-emerald-400">Finance</span>
        </h1>
        <p className="mt-2 text-slate-400">
          Multi-Investor RAG Agent — grounded advice, cited to the source.
        </p>

        <div className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            System status
          </h2>
          {error && (
            <p className="rounded-lg border border-rose-800 bg-rose-950/50 p-4 text-sm text-rose-300">
              Backend unreachable: {error} — is uvicorn running on :8000?
            </p>
          )}
          {health && (
            <div className="grid gap-2">
              <StatusPill label="MongoDB" ok={health.mongo === 'up'} />
              <StatusPill label="ChromaDB" ok={health.chroma === 'up'} />
              <StatusPill
                label={`LLM (${health.llm_model})`}
                ok={health.llm_configured}
              />
            </div>
          )}
          {!health && !error && <p className="text-slate-500">Checking…</p>}
        </div>

        <p className="mt-10 text-xs text-slate-600">
          Checkpoint 1 — foundations. Chat UI arrives at Checkpoint 3.
        </p>
      </div>
    </div>
  )
}
