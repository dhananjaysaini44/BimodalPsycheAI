# Repository Structure — Multimodal Depression Detection System

A **monorepo** with four top-level services, matching your architecture exactly:
`frontend` (React/Next.js) → `backend` (Node/Express, API gateway + security) → `ml-engine` (FastAPI, serving) — plus `ml-training`, kept **separate from serving** since it's an offline, one-time (or periodic) job, not something that runs per-request.

```
depression-detection-system/
│
├── frontend/                          # React / Next.js
│   ├── app/                           # routes: /auth, /consent, /chat, /results, /history
│   ├── components/
│   │   ├── auth/                      # login screen, Firebase wiring
│   │   ├── consent/                   # disclaimer screen
│   │   ├── chat/                      # conversational UI, resume banner, loading state
│   │   ├── results/                   # results dashboard, explanation display
│   │   ├── crisis/                    # crisis resource panel (interrupt + high-severity)
│   │   └── history/                   # past assessments view
│   ├── lib/
│   │   ├── firebaseClient.js
│   │   └── apiClient.js               # wraps /session/* calls
│   ├── public/
│   ├── .env.local.example             # Firebase config keys (placeholders only)
│   └── package.json
│
├── backend/                           # Node.js / Express — API gateway + security layer
│   ├── src/
│   │   ├── routes/
│   │   │   ├── auth.routes.js
│   │   │   ├── session.routes.js      # /session/start, /turn, /complete, /draft, /resume/:id
│   │   │   └── history.routes.js
│   │   ├── controllers/
│   │   ├── middleware/
│   │   │   ├── verifyFirebaseToken.js
│   │   │   ├── validateInput.js       # size/format/length checks
│   │   │   └── errorHandler.js        # timeout + failure handling on ML calls
│   │   ├── services/
│   │   │   ├── mlServiceClient.js     # calls the FastAPI ml-engine
│   │   │   └── encryption.service.js  # AES-256-GCM encrypt/decrypt
│   │   ├── db/
│   │   │   ├── schema.sql             # users, assessments, sessions tables
│   │   │   └── migrations/
│   │   ├── config/
│   │   └── server.js
│   ├── .env.example                   # ENCRYPTION_KEY, FIREBASE_*, ML_ENGINE_URL (placeholders only)
│   └── package.json
│
├── ml-engine/                         # Python FastAPI — model SERVING only
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── orchestrator.py        # per-turn: sentiment + crisis check + next question
│   │   │   ├── text_analysis.py       # session-end
│   │   │   ├── voice_analysis.py      # session-end
│   │   │   └── fusion.py
│   │   ├── orchestrator/
│   │   │   ├── sentiment_scorer.py    # fast model (VADER / lightweight LogReg)
│   │   │   ├── crisis_detector.py     # keyword/pattern check
│   │   │   └── question_selector.py   # reads question_bank.json
│   │   ├── text_module/
│   │   │   ├── preprocess.py
│   │   │   ├── features.py
│   │   │   └── classifier.py          # loads trained model artifact at startup
│   │   ├── voice_module/
│   │   │   ├── preprocess.py
│   │   │   ├── features.py            # librosa: MFCC, pitch, energy, speech rate
│   │   │   └── classifier.py
│   │   ├── fusion/
│   │   │   ├── fuse.py                # weighted average → severity
│   │   │   └── explain.py             # merges top signals into plain-language summary
│   │   ├── schemas/                   # pydantic request/response models
│   │   └── core/config.py
│   ├── models/                        # trained artifacts — GITIGNORED, not committed
│   │   └── README.md                  # explains where to fetch/place model files
│   ├── data/
│   │   └── question_bank.json         # 7 themes × light/deep phrasing variants
│   └── requirements.txt
│
├── ml-training/                       # OFFLINE pipeline — kept separate from serving
│   ├── notebooks/                     # EDA, experiments
│   ├── scripts/
│   │   ├── preprocess_edaic.py
│   │   ├── train_text_model.py
│   │   ├── train_voice_model.py
│   │   └── evaluate.py                # compare against Al Hanai et al. baseline
│   ├── datasets/                      # GITIGNORED — E-DAIC/MDDInterview raw data
│   │   └── README.md                  # how to request/place data, NOT the data itself
│   ├── artifacts/                     # exported models, versioned e.g. edaic_v1_bert.pt
│   └── requirements.txt
│
├── docs/
│   ├── synopsis/                      # SYNOPSIS_FINAL.pdf
│   ├── diagrams/
│   │   ├── architecture.mermaid
│   │   ├── dfd_level0.mermaid
│   │   ├── dfd_level1.mermaid
│   │   └── sequence_diagram.mermaid
│   └── data-use-agreement/            # your signed E-DAIC agreement (NOT the dataset)
│
├── .gitignore
├── docker-compose.yml                 # optional: local dev — backend + ml-engine together
├── README.md
└── LICENSE
```

## Why split this way

- **`ml-engine` vs `ml-training` are separate folders, not just separate scripts.** This mirrors the distinction we made earlier: `ml-engine` is what runs in production per-request (fast, serving-only, loads a finished model); `ml-training` is what you run occasionally on your own machine against E-DAIC to *produce* that model. Mixing them makes it unclear what actually needs to be deployed to Render vs. what's a one-time local job — keep the model artifact as the only thing that crosses from one folder to the other.
- **`backend/src/services/encryption.service.js` is its own file**, not inlined into routes — matches the "Security Layer" being a distinct component in your architecture diagram, and means every place that touches storage/DB goes through one audited path.
- **`ml-engine/app/orchestrator/` vs the `text_module/`/`voice_module/` folders** enforces the fast/slow split you designed — a new contributor (or you, in six months) can see immediately which code runs per-turn and which runs once per session.
- **`data/question_bank.json`** lives in the serving engine (it's read live), while **`ml-training/datasets/`** never gets committed — the E-DAIC data-use agreement explicitly forbids redistribution, so this folder should exist only as a placeholder with instructions, never actual data files.

## `.gitignore` — non-negotiable entries given your data

```gitignore
# Environment / secrets
.env
.env.local
*.env

# Databases
*.db
*.sqlite

# Datasets — E-DAIC/MDDInterview must never be committed (data-use agreement)
ml-training/datasets/*
!ml-training/datasets/README.md

# Trained model artifacts — too large for git, and may embed dataset traces
ml-engine/models/*.pkl
ml-engine/models/*.pt
ml-engine/models/*.h5
!ml-engine/models/README.md

# Audio files (encrypted at rest, but shouldn't be in git regardless)
backend/uploads/
storage/

node_modules/
__pycache__/
*.pyc
```

If you need model files or E-DAIC-derived training data shared with teammates, use something outside git entirely — a shared drive link, or Git LFS with private access only — never a public commit, since that would breach the data-use agreement you're signing.

---

## Starter Files

Minimal, working skeletons for each service — enough to `git init`, install dependencies, and start building on top of.

### `frontend/package.json`

```json
{
  "name": "depression-detection-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "firebase": "^10.12.0",
    "axios": "^1.7.0"
  },
  "devDependencies": {
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^8.57.0",
    "eslint-config-next": "^14.2.0"
  }
}
```

### `frontend/.env.local.example`

```bash
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000
```

### `frontend/lib/apiClient.js`

```javascript
import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL,
});

export const startSession = (idToken) =>
  api.post("/session/start", {}, { headers: { Authorization: `Bearer ${idToken}` } });

export const submitTurn = (sessionId, payload, idToken) =>
  api.post(`/session/turn`, { sessionId, ...payload }, {
    headers: { Authorization: `Bearer ${idToken}` },
  });

export const completeSession = (sessionId, idToken) =>
  api.post(`/session/complete`, { sessionId }, {
    headers: { Authorization: `Bearer ${idToken}` },
  });

export const getHistory = (idToken) =>
  api.get("/history", { headers: { Authorization: `Bearer ${idToken}` } });

export default api;
```

---

### `backend/package.json`

```json
{
  "name": "depression-detection-backend",
  "version": "0.1.0",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  },
  "dependencies": {
    "express": "^4.19.0",
    "better-sqlite3": "^11.1.0",
    "firebase-admin": "^12.2.0",
    "axios": "^1.7.0",
    "multer": "^1.4.5-lts.1",
    "dotenv": "^16.4.0",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "nodemon": "^3.1.0"
  }
}
```

### `backend/.env.example`

```bash
PORT=5000
ML_ENGINE_URL=http://localhost:8000
ENCRYPTION_KEY=          # 32-byte key, generate with: openssl rand -hex 32
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
DATABASE_PATH=./src/db/app.db
```

### `backend/src/server.js`

```javascript
require("dotenv").config();
const express = require("express");
const cors = require("cors");

const authRoutes = require("./routes/auth.routes");
const sessionRoutes = require("./routes/session.routes");
const historyRoutes = require("./routes/history.routes");
const errorHandler = require("./middleware/errorHandler");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/auth", authRoutes);
app.use("/session", sessionRoutes);
app.use("/history", historyRoutes);

app.get("/health", (req, res) => res.json({ status: "ok" }));

app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
```

### `backend/src/middleware/verifyFirebaseToken.js`

```javascript
const admin = require("firebase-admin");
const serviceAccount = require(process.env.FIREBASE_SERVICE_ACCOUNT_PATH);

if (!admin.apps.length) {
  admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
}

module.exports = async function verifyFirebaseToken(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;

  if (!token) return res.status(401).json({ error: "Missing auth token" });

  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.userId = decoded.uid;
    next();
  } catch (err) {
    res.status(401).json({ error: "Invalid or expired token" });
  }
};
```

### `backend/src/services/encryption.service.js`

```javascript
const crypto = require("crypto");

const ALGORITHM = "aes-256-gcm";
const KEY = Buffer.from(process.env.ENCRYPTION_KEY, "hex"); // 32 bytes

function encrypt(plainText) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv(ALGORITHM, KEY, iv);
  const encrypted = Buffer.concat([cipher.update(plainText, "utf8"), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return Buffer.concat([iv, authTag, encrypted]).toString("base64");
}

function decrypt(payload) {
  const data = Buffer.from(payload, "base64");
  const iv = data.subarray(0, 12);
  const authTag = data.subarray(12, 28);
  const encrypted = data.subarray(28);
  const decipher = crypto.createDecipheriv(ALGORITHM, KEY, iv);
  decipher.setAuthTag(authTag);
  return Buffer.concat([decipher.update(encrypted), decipher.final()]).toString("utf8");
}

module.exports = { encrypt, decrypt };
```

### `backend/src/db/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,           -- Firebase UID
  email TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  status TEXT CHECK(status IN ('draft', 'complete')) DEFAULT 'draft',
  turn_history TEXT,             -- encrypted JSON blob
  sentiment_trend TEXT,
  current_theme TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS assessments (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  score REAL,
  severity TEXT CHECK(severity IN ('Low', 'Moderate', 'High')),
  explanation TEXT,              -- encrypted
  model_version TEXT,
  audio_refs TEXT,               -- JSON array of encrypted file paths
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (session_id) REFERENCES conversation_sessions(id)
);
```

---

### `ml-engine/requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
pydantic==2.7.0
librosa==0.10.2
numpy==1.26.4
scikit-learn==1.5.0
vaderSentiment==3.3.2
soundfile==0.12.1
```

### `ml-engine/app/main.py`

```python
from fastapi import FastAPI
from app.routers import orchestrator, text_analysis, voice_analysis, fusion

app = FastAPI(title="Depression Detection ML Engine")

app.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestrator"])
app.include_router(text_analysis.router, prefix="/analyze/text", tags=["text"])
app.include_router(voice_analysis.router, prefix="/analyze/voice", tags=["voice"])
app.include_router(fusion.router, prefix="/fusion", tags=["fusion"])


@app.get("/health")
def health():
    return {"status": "ok"}
```

### `ml-engine/app/orchestrator/sentiment_scorer.py`

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def score_turn_sentiment(text: str) -> float:
    """Fast, lightweight sentiment score for steering the conversation.
    NOT the final depression classifier — that runs once at session end."""
    return analyzer.polarity_scores(text)["compound"]  # -1 (negative) to +1 (positive)
```

### `ml-engine/app/orchestrator/crisis_detector.py`

```python
# NOTE: keep this list in a config file, not hardcoded, so it can be
# reviewed/updated without a code change. This is a known-limitation
# component — pattern matching alone will miss indirect risk signals.
RISK_PATTERNS = [
    # populate with reviewed, clinically-informed risk phrases
]


def check_crisis_signal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in RISK_PATTERNS)
```

### `ml-engine/data/question_bank.json`

```json
{
  "themes": [
    {
      "id": "mood",
      "light": "How's your week been so far?",
      "deep": "It sounds like things have been heavy lately — want to tell me more about that?"
    },
    {
      "id": "sleep",
      "light": "How have you been sleeping?",
      "deep": "Has your sleep been affected by how you've been feeling?"
    },
    {
      "id": "energy",
      "light": "How's your energy been lately?",
      "deep": "Have you noticed it's been harder to get through the day?"
    },
    {
      "id": "social",
      "light": "Have you been spending time with friends or family?",
      "deep": "Have you found yourself pulling away from people lately?"
    },
    {
      "id": "concentration",
      "light": "How's it been to focus on things — work, study, hobbies?",
      "deep": "Has it felt harder than usual to concentrate on things?"
    },
    {
      "id": "self_worth",
      "light": "How have you been feeling about yourself lately?",
      "deep": "Have you been feeling down on yourself more than usual?"
    },
    {
      "id": "outlook",
      "light": "How are you feeling about things going forward?",
      "deep": "Has it felt hard to see things getting better lately?"
    }
  ]
}
```

---

### `ml-training/requirements.txt`

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
librosa==0.10.2
torch==2.3.0
transformers==4.42.0
matplotlib==3.9.0
jupyter==1.0.0
```

### `ml-training/datasets/README.md`

```markdown
# Datasets — DO NOT COMMIT DATA HERE

This folder is for local use only. E-DAIC is distributed under a
data-use agreement with USC ICT that forbids redistribution.

## Setup
1. Request access: https://dcapswoz.ict.usc.edu/
2. Once approved, place files here as:
   datasets/edaic/train/, datasets/edaic/dev/, datasets/edaic/test/
3. For pipeline development before approval comes through, use
   MDDInterview (openly hosted): github.com/uofabinarylab/MDDInterview
```

---

### `docker-compose.yml`

```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    env_file: ./backend/.env
    depends_on:
      - ml-engine

  ml-engine:
    build: ./ml-engine
    ports:
      - "8000:8000"
```

### `README.md`

```markdown
# Multimodal Depression Detection System

Bimodal (voice + text) depression screening via an adaptive,
conversational assessment. See `docs/diagrams/` for architecture,
DFD, and sequence diagrams.

## Services
- `frontend/` — Next.js chat UI
- `backend/` — Express API gateway (auth, encryption, orchestration)
- `ml-engine/` — FastAPI model serving (orchestrator + text/voice/fusion)
- `ml-training/` — offline training pipeline (not deployed)

## Local setup
1. `cd backend && npm install && cp .env.example .env`
2. `cd ml-engine && pip install -r requirements.txt`
3. `cd frontend && npm install && cp .env.local.example .env.local`
4. `docker-compose up` (backend + ml-engine), then `npm run dev` in `frontend/`

## Dataset
Trained on E-DAIC (USC ICT) — see `ml-training/datasets/README.md`
for access instructions. Dataset itself is never committed to this repo.
```
