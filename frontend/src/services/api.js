// Centralized Axios service layer — all backend calls live here.
// Endpoints follow API_CONTRACT.md; add functions as checkpoints land.
import axios from 'axios'

const api = axios.create({
  // Relative paths — Vite dev server proxies /api and /health to :8000
  timeout: 60000,
})

// --- Health ---
export const getHealth = () => api.get('/health').then((r) => r.data)
export const getComponentHealth = () =>
  api.get('/health/components').then((r) => r.data)

// --- Chat (Checkpoint 3) ---
export const sendChat = (query, persona = 'auto', sessionId = null) =>
  api
    .post('/api/v1/chat', { query, persona, session_id: sessionId })
    .then((r) => r.data)

// --- Personas (Checkpoint 4) ---
export const getPersonas = () => api.get('/api/v1/personas').then((r) => r.data)

// --- Market (Checkpoint 5) ---
export const getQuote = (symbol) =>
  api.get(`/api/v1/market/quote/${symbol}`).then((r) => r.data)

export default api
