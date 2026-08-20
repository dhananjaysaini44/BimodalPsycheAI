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
