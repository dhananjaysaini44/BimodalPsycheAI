# PsycheAI Frontend

Next.js web client for registration, authentication, account settings, user history, and assessment sessions. Product-facing Terms of Service and Privacy Policy pages are available at `/terms` and `/privacy` and are linked from registration.

## Run locally

Requirements: Node.js and the backend's required services. Create `frontend/.env.local` with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8080
```

For browser-based local development, configure the backend with `CORS_ALLOWED_ORIGINS=http://localhost:3000` and `COOKIE_SECURE=false` (HTTP development only). Start MySQL and the backend first, then from `frontend`:

```powershell
npm run dev
```

Open <http://localhost:3000>. Register with email OTP and accept both the Terms of Service and Privacy Policy. Sign in, complete and save both history forms, then start or resume an assessment from the dashboard. The assessment page retrieves the next question from the backend and supports text responses or audio-file uploads; it does not capture live microphone audio. Do not manually set multipart `Content-Type`; the browser supplies its boundary. Use **Account** from the dashboard to edit profile information (email is immutable), request a verification code, change the password, or delete the account. Password changes and deletion require the current password and a single-use OTP sent to the registered email.

Text submissions require the ML service's `/orchestrator/sentiment` and `/orchestrator/crisis` routes. Audio submissions are stored encrypted at rest by the backend under `audio_files/{sessionId}/{turnId}/audio.enc` by default and display `PENDING_MODEL_ANALYSIS`. The ML `/analyze/text/`, `/analyze/voice/`, and `/fusion/` endpoints are currently placeholders, so final assessment generation may return HTTP 503; the page reports that failure and does not display fabricated results. Screening output is not a diagnosis or substitute for professional care.

## Checks

```powershell
npm run lint
npm run build
```
