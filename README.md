# Multimodal Depression Detection System

Bimodal (voice + text) depression screening via an adaptive,
conversational assessment. See `docs/diagrams/` for architecture,
DFD, and sequence diagrams.

## Services
- `frontend/` — Next.js chat UI (TypeScript)
- `backend/` — Express API gateway (auth, encryption, orchestration in TypeScript)
- `ml-engine/` — FastAPI model serving (orchestrator + text/voice/fusion in Python)
- `ml-training/` — offline training pipeline (Python, not deployed)

## Local setup
1. `cd backend && npm install && cp .env.example .env`
2. `cd ml-engine && pip install -r requirements.txt`
3. `cd frontend && npm install && cp .env.local.example .env.local`
4. `docker-compose up` (backend + ml-engine), then `npm run dev` in `frontend/`

## Dataset
Trained on E-DAIC (USC ICT) — see `ml-training/datasets/README.md`
for access instructions. Dataset itself is never committed to this repo.
