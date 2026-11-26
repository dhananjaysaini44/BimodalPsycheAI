# BiModalPsyche Backend

Spring Boot REST API for account registration, email verification, JWT authentication, and assessment-session lifecycle management.

## Current API scope

Implemented HTTP endpoints:

| Method | Path | Authentication | Purpose |
|---|---|---|---|
| `POST` | `/auth/send-otp` | Public | Send an email verification code. |
| `POST` | `/auth/verify-otp` | Public | Verify the code sent to an email. |
| `POST` | `/auth/register` | Public | Create an account after email verification and consent. |
| `POST` | `/auth/login` | Public | Authenticate and issue access and refresh tokens. |
| `POST` | `/auth/refresh` | Refresh cookie | Rotate the refresh token and issue a new access token. |
| `POST` | `/auth/logout` | Refresh cookie optional | Revoke the current refresh session and clear its cookie. |
| `POST` | `/session/start` | Access JWT | Create a new assessment session. |
| `GET` | `/session/draft` | Access JWT | Find the user's unfinished session. |
| `GET` | `/session/{sessionId}` | Access JWT | Fetch one session owned by the authenticated user. |
| `POST` | `/session/resume/{sessionId}` | Access JWT | Resume an unfinished session. |
| `POST` | `/session/complete/{sessionId}` | Access JWT | Complete an active or paused session. |
| `POST` | `/session/abandon/{sessionId}` | Access JWT | Abandon an unfinished session while retaining its record. |
| `GET` | `/questions/active` | Access JWT | Fetch the active question bank and its variants. |
| `GET` | `/session/{sessionId}/question/next` | Access JWT | Get the next question selected for an owned session. |
| `POST` | `/session/{sessionId}/turns` | Access JWT | Analyze and store one text response in an owned, in-progress session. |
| `POST` | `/session/{sessionId}/turns/audio` | Access JWT | Encrypt and store one audio response for the expected question. |
| `GET` | `/session/{sessionId}/turns` | Access JWT | Read the owner's session transcript, decrypting answers server-side. |
| `POST` | `/session/{sessionId}/assessment` | Access JWT | Generate and persist final text, voice, or fused assessment results. |
| `GET` | `/session/{sessionId}/assessment` | Access JWT | Read a previously generated assessment result. |
| `GET` | `/user/profile` | Access JWT | Read the authenticated user's profile; email is returned but immutable. |
| `PUT` | `/user/profile` | Access JWT | Update editable profile and demographic fields without changing email or consent/security state. |
| `PUT` | `/user/password` | Access JWT | Change the password after verifying the current password; revokes refresh sessions. |
| `DELETE` | `/user/account` | Access JWT | Permanently delete the account and its associated data after password confirmation. |
| `GET` | `/user/history/medical` | Access JWT | Get the authenticated user's medical-history details and completion state. |
| `PUT` | `/user/history/medical` | Access JWT | Create or update the authenticated user's medical history. |
| `GET` | `/user/history/family` | Access JWT | Get the authenticated user's family-history details and completion state. |
| `PUT` | `/user/history/family` | Access JWT | Create or update the authenticated user's family history. |

Audio submission and final assessment-result endpoints are implemented. The backend encrypts audio at rest and defers model analysis until final assessment generation. Question progression is server-selected; it chooses a deeper variant after a negative text sentiment score (at or below -0.25), otherwise a light variant, and asks each theme in the session's pinned question-bank version once. Text analysis uses the working sentiment/crisis routes. Final text, voice, and fusion analysis requires ML routes that are currently placeholders; those calls fail with HTTP 503 rather than returning fabricated results.

## Requirements and configuration

- Java 27 and the included Maven wrapper (`mvnw.cmd` on Windows).
- MySQL database, configured with `DB_URL`, `MYSQLUSER`, and `MYSQLPASSWORD`.
- Email delivery configured with `MAIL_USERNAME` and `MAIL_PASSWORD`; send-OTP requires a functioning mail provider.
- `JWT_SECRET` must be Base64-encoded key material of at least 256 bits. Provide it through the environment; do not commit it.
- `ENCRYPTION_KEY` must be Base64-encoded 32-byte AES key material. Generate a unique key for the environment and keep it outside source control; encrypted session responses cannot be recovered if this key is lost.
- `ML_ENGINE_BASE_URL` defaults to `http://localhost:8000`; point it at the ML service reachable by the backend.
- Configure `CORS_ALLOWED_ORIGINS` as a comma-separated list of trusted browser frontend origins.
- `COOKIE_SECURE` defaults to `true`; use HTTPS and leave it enabled in production. For local HTTP-only development, it may be set to `false`.
- `TERMS_VERSION` and `PRIVACY_POLICY_VERSION` default to `v1`.
- `QUESTION_BANK_VERSION` selects the version seeded on a new database; sessions store the active version when they start.
- `AUDIO_STORAGE_DIR` defaults to `./data/audio`; keep this directory private and backed up according to your retention policy. Uploaded audio is encrypted before it is written.
- `AUDIO_MAX_SIZE_BYTES` defaults to `26214400` (25 MiB); Spring multipart limits are configured to 25 MiB per file and 26 MiB per request.

The application uses `spring.jpa.hibernate.ddl-auto=update` to update the configured database schema. Back up and review production database changes before deploying schema updates.

Generate a development encryption key in PowerShell with:

```powershell
[Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

### Database migrations

Hibernate's schema update does not reliably add values to an existing MySQL `ENUM` column. Apply checked-in SQL migrations explicitly when upgrading an existing database. For the session-abandon feature, run `backend/db/migrations/20261022_add_abandoned_session_status.sql` against the application database before calling `POST /session/abandon/{sessionId}`. The migration preserves existing statuses and adds `ABANDONED`.

## Authentication and token behavior

The only public routes are the listed `POST /auth/...` operations. Every other route requires an access JWT:

```http
Authorization: Bearer <access-token>
```

Unauthenticated protected-route requests return HTTP 401. The backend validates the token; frontend route guards are not an authorization boundary. Session reads and mutations are owner-scoped: requests for another account's session return HTTP 404.

Login and refresh return JSON shaped like:

```json
{
  "message": "Login Successful",
  "token": "<access-token>"
}
```

The actual access-token value is a signed JWT. The refresh token is not returned in the JSON body; it is set in an `HttpOnly` cookie named `refreshToken`, scoped to `/auth`, with `SameSite=Strict`, a seven-day lifetime, and the `Secure` flag controlled by `COOKIE_SECURE`.

Access tokens last 15 minutes. The refresh token is rotated on each successful refresh and stored server-side only as a SHA-256 hash. A refresh-row lock serializes rotation. Reuse of an already-rotated token revokes the current refresh session. Access tokens are bound to the refresh-session row and token version; each protected API request checks that the session is still active, so logout and a subsequent login invalidate earlier access tokens immediately. This check requires a database lookup on authenticated requests.

### Profile, password, and account deletion

`GET /user/profile` returns the authenticated account's profile, including its email address. `PUT /user/profile` accepts the editable fields `firstName`, `lastName`, `dateOfBirth` (`YYYY-MM-DD`), `gender`, `maritalStatus`, `jobTitle`, `employmentType`, `phoneNumber`, and `workingHoursPerWeek`. The request is a full profile update: required demographic fields must be provided, and nullable fields can be set to `null` to clear them. The API does not accept email, password, role, verification, legal-consent, or server-managed completion fields for profile updates.

```json
{
  "firstName": "Jordan",
  "lastName": "Lee",
  "dateOfBirth": "1990-04-12",
  "gender": "OTHERS",
  "maritalStatus": "SINGLE",
  "jobTitle": "Designer",
  "employmentType": "EMPLOYED",
  "phoneNumber": "+15551234567",
  "workingHoursPerWeek": 40
}
```

The enum values are `gender`: `MALE`, `FEMALE`, `OTHERS`; `maritalStatus`: `SINGLE`, `MARRIED`, `DIVORCED`, `WIDOWED`; `employmentType`: `STUDENT`, `SELF_EMPLOYED`, `EMPLOYED`, `UNEMPLOYED`, `OTHERS`, `RETIRED`. A successful profile update returns HTTP 200 with the saved profile; invalid fields return HTTP 400.

Before changing the password or deleting the account, request a one-time email code with `POST /user/security/send-otp`. The authenticated account's email receives a six-digit code. Codes expire after five minutes, have a 60-second resend cooldown, and are single-use; failed verification attempts are limited.

To change the password, call `PUT /user/password` with the current password, new password, and email code:

```json
{
  "currentPassword": "current-password",
  "newPassword": "new-password-at-least-8-characters",
  "otp": "123456"
}
```

The current password and a valid email code are both required. A successful password change invalidates the account's refresh session; sign in again with the new password.

To permanently delete an account, call `DELETE /user/account` with:

```json
{
  "currentPassword": "current-password",
  "otp": "123456"
}
```

The backend removes the account's assessment history, session turns and analyses, crisis events, encrypted audio metadata, medical/family histories, OTP record, and refresh session in one database transaction. The account's audio-storage directory (including orphan files) is removed after the database transaction commits. On success the endpoint returns HTTP 204 and expires the refresh cookie. Audio-file cleanup failures are logged for operational follow-up; they do not restore the already deleted account. This deletion is permanent and cannot be undone.

The frontend keeps access tokens in JavaScript memory only; they are not written to `localStorage` or `sessionStorage` and are cleared on logout. A full page reload obtains a fresh access token through the `HttpOnly` refresh cookie. Protected API calls also attempt one refresh when the access token is missing or receives HTTP 401, then retry the original request once. Concurrent refresh attempts within the same page share one in-flight refresh request. Postman can retain the refresh cookie in its cookie jar; do not paste real credentials or token values into shared collections or source control.

Memory-only access-token storage reduces exposure to persistent token theft from browser storage, but it does not prevent an active same-origin XSS from issuing authenticated requests while a user is signed in. Continue to avoid unsafe HTML/script injection and keep frontend dependencies maintained.

## Registration consent

Account creation requires both the current Terms of Service and Privacy Policy to be accepted. The server rejects either value missing or `false`; successful registration stores each document version and acceptance timestamp. The current values are configured via `TERMS_VERSION` and `PRIVACY_POLICY_VERSION`. Updating a document requires deploying its new version value. Existing users are not automatically considered to have accepted a new version, so starting or resuming a session is blocked if their stored versions are outdated. An endpoint for existing users to renew consent is not implemented yet.

Email verification codes expire after five minutes, have a 60-second resend cooldown, and are invalidated after five incorrect verification attempts. Verification and resend operations lock an existing OTP record while updating it, so concurrent requests cannot bypass the attempt counter or overwrite one another's active code.

## Medical and family history prerequisites

Before starting an assessment session, the authenticated user must submit both a medical-history form and a family-history form. These are self-reported, free-text fields with a maximum of 10,000 characters each. Users with no known history should explicitly enter `None known`; blank submissions are rejected. Only the user's own records are read or updated. The backend sets the completion flags when each form is saved; clients cannot set completion by sending a boolean.

The profile endpoints are:

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/user/history/medical` | Returns `{ "details": null, "completed": false }` until saved, otherwise the stored details and `completed: true`. |
| `PUT` | `/user/history/medical` | Creates or updates medical history and marks it complete. |
| `GET` | `/user/history/family` | Returns `{ "details": null, "completed": false }` until saved, otherwise the stored details and `completed: true`. |
| `PUT` | `/user/history/family` | Creates or updates family history and marks it complete. |

`POST /session/start` returns HTTP 409 until both forms are complete. `POST /session/resume/{sessionId}` does not re-gate history; existing sessions can be resumed. History is sensitive personal information: avoid collecting information not needed by the experience and do not place real medical/family details in shared Postman collections or logs.

## OTP registration workflow

Complete these requests in order:

1. `POST /auth/send-otp` sends a six-digit OTP to the supplied email. Codes expire after five minutes; resending is rate-limited by a 60-second cooldown.
2. `POST /auth/verify-otp` verifies the actual code delivered to that inbox. Five failed attempts invalidate the code. Successful verification opens a 15-minute registration window.
3. `POST /auth/register` creates the account during that window. Both legal-consent booleans must be `true`.
4. `POST /auth/login` returns the access JWT and sets the refresh cookie.
5. Save both authenticated history forms below before starting the first session.

The OTP must come from the email received during testing. There is no fixed dummy OTP or OTP in the API response.

## Assessment session lifecycle

Starting creates an `IN_PROGRESS` session immediately and pins the currently active question-bank version to it. Only one unfinished session (`DRAFT`, `IN_PROGRESS`, or `PAUSED`) is allowed per user. A second start returns HTTP 409 until that unfinished session is resumed or abandoned. Starting and resuming require acceptance timestamps and versions matching the currently configured Terms and Privacy Policy. Starting additionally requires the medical and family history completion flags. Completing requires one answer per theme in the session's pinned bank; changing the active bank does not change the required prompts for existing sessions. The current seeded bank has seven themes. Abandoning does not require current consent or profile completion, so the user can close an existing session.

Allowed lifecycle behavior:

| Operation | Accepted state | Result |
|---|---|---|
| Start | No unfinished session | Creates `IN_PROGRESS` (HTTP 201). |
| Find draft | Any | Returns the newest unfinished session (HTTP 200), or no body (HTTP 204). |
| Get session | Any existing owned session | Returns session data (HTTP 200). |
| Resume | `DRAFT`, `PAUSED`, or `IN_PROGRESS` | Sets/resets activity and returns `IN_PROGRESS` (HTTP 200). `IN_PROGRESS` is effectively idempotent. |
| Complete | `IN_PROGRESS` or `PAUSED`, all active themes answered | Sets `COMPLETED` and `completedAt` (HTTP 200); otherwise HTTP 409. |
| Abandon | `DRAFT`, `IN_PROGRESS`, or `PAUSED` | Sets `ABANDONED` and retains the record (HTTP 200). |

There is currently no pause operation or API which creates a `DRAFT` session. The resume endpoint can nevertheless be exercised against an `IN_PROGRESS` session created by `/session/start`.

### Question bank and progression

`GET /questions/active` returns the currently active bank version and question variants. `GET /session/{sessionId}/question/next` returns the next unanswered theme and the server-selected phrasing from that session's pinned version, or `conversationComplete: true` when all pinned themes have been answered. A paused session returns `available: false`; resume it explicitly to continue. The backend initializes missing seeded themes/variants on application startup without overwriting existing question text. The initial content mirrors the current seven themes in `ml-engine/data/question_bank.json`. Set `QUESTION_BANK_VERSION` when bootstrapping a new database; an already active bank is used. Legacy sessions without a pinned version continue using the active bank.

### Text-turn submission

`POST /session/{sessionId}/turns` accepts text only. It requires an authenticated owner of an `IN_PROGRESS` session and the ID of the exact next question returned by the backend. The backend calls the ML engine's `/orchestrator/sentiment` and `/orchestrator/crisis` routes, encrypts the submitted response using AES-256-GCM before database persistence, and stores the per-turn outputs in `session_turns` and `turn_analysis`. The database value is version-prefixed (`v1:`) and includes a fresh nonce for each encryption. Medical and family history are encrypted with the same key; legacy unprefixed history values are re-encrypted on the first read.

The endpoint returns the question snapshot, turn number, sentiment value, crisis flag, generic guidance, matching configured crisis resources, and timestamp; it never returns the submitted answer. A failed or invalid ML response returns HTTP 503 and does not save the turn. When the crisis flag is raised, the event is recorded and the session is paused; the response includes non-diagnostic guidance and active rows in `crisis_resources`. On startup, the backend seeds India's Tele-MANAS helpline (14416) if it is not already present; the resource record links to the [official Tele-MANAS website](https://telemanas.mohfw.gov.in/). Add and verify other country-specific resources before relying on them for users elsewhere. The current sentiment is VADER and crisis detection is a limited phrase matcher; these outputs are not a trained classifier, clinical assessment, or safety guarantee. No final score, severity, explanation, or diagnosis is produced.

`GET /session/{sessionId}/turns` returns the authenticated owner's turns in order, including decrypted response text. Requests for another user's session return 404. Treat these responses as sensitive health information; do not log or cache them in shared systems.

### Audio-turn submission and final assessment

`POST /session/{sessionId}/turns/audio` consumes `multipart/form-data` with a `file` part and a `questionVariantId` field. It applies the same ownership, in-progress-session, and expected-question checks as text turns. Accepted formats are WebM, WAV, MP3, MP4, OGG, and AAC (maximum 25 MiB, configurable with `AUDIO_MAX_SIZE_BYTES`). Audio bytes are encrypted with AES-256-GCM and stored at the repository root in `audio_files/{sessionId}/{turnId}/audio.enc` by default when the backend is launched from its directory (configurable with `AUDIO_STORAGE_DIR`); the database contains file metadata and a relative storage path, not the audio bytes. The response's `analysisStatus` is `PENDING_MODEL_ANALYSIS`; voice analysis is deferred until the session is completed.

After answering all themes, call `POST /session/complete/{sessionId}`, then `POST /session/{sessionId}/assessment`. The backend analyzes available text turns through `/analyze/text/`, audio through `/analyze/voice/`, and calls `/fusion/` when both modalities are present. Single-modality sessions use that modality's result. It stores modality-level scores and the final result only after all required ML calls return valid data. `GET /session/{sessionId}/assessment` reads the saved result. Results include `score`, `probability`, `severity` (`LOW`, `MODERATE`, or `HIGH`), an optional explanation, and per-modality entries. The ML routes currently return placeholders, so assessment generation responds HTTP 503 and saves no result until the ML service implements the contract. Text-turn sentiment/crisis analysis remains available independently.

The ML service receives the response text to perform its analysis. Keep the service on a trusted network; the text is encrypted at rest by the backend, but is sent to the configured ML service for processing.

Session response shape:

```json
{
  "id": 123,
  "status": "IN_PROGRESS",
  "currentTurnNumber": 0,
  "startedAt": "2026-10-22T10:00:00",
  "lastActivityAt": "2026-10-22T10:00:00",
  "completedAt": null,
  "createdAt": "2026-10-22T10:00:00"
}
```

Timestamps are examples; actual values are generated by the server.

## Postman quick start

### Environment variables

Create a local Postman environment with:

| Variable | Example value |
|---|---|
| `baseUrl` | `http://localhost:8080` |
| `testEmail` | `psyche.test@example.com` |
| `testPassword` | `LocalTestPass!2026` |
| `accessToken` | Leave empty; login/refresh scripts set it. |
| `sessionId` | Leave empty; the session-start script sets it. |
| `questionVariantId` | Set from the next-question response script below. |

Use an inbox you control for `testEmail`. These are dummy test values only; use a unique address not already registered in the database. Keep secrets and real tokens in a private local Postman environment, not an exported/shared collection.

### Common Postman setup

- Set request body type to **raw → JSON** for requests with a body.
- For session requests add `Authorization: Bearer {{accessToken}}`.
- The refresh cookie is set by login and rotated by refresh. Postman normally stores and sends it automatically for the same host. If it does not, enable cookie handling for `localhost`; do not put the refresh token in a JSON body.
- For protected-request negative testing, omit the Authorization header; expect HTTP 401.

### 1. Send OTP

`POST {{baseUrl}}/auth/send-otp`

```json
{
  "email": "{{testEmail}}"
}
```

Expected: HTTP 200 and `"Otp Sent Successfully"`. Retrieve the OTP from the test inbox. A second send within the cooldown returns HTTP 429.

### 2. Verify OTP

`POST {{baseUrl}}/auth/verify-otp`

```json
{
  "email": "{{testEmail}}",
  "otp": "123456"
}
```

Replace `123456` with the actual six-digit email code. Expected: HTTP 200 and `"OTP verified successfully"`.

### 3. Register

`POST {{baseUrl}}/auth/register`

```json
{
  "email": "{{testEmail}}",
  "password": "{{testPassword}}",
  "firstName": "Psyche",
  "lastName": "Test",
  "gender": "OTHERS",
  "maritalStatus": "SINGLE",
  "jobTitle": "Student",
  "employmentType": "STUDENT",
  "phoneNumber": "+15555550123",
  "workingHoursPerWeek": 10,
  "dateOfBirth": "2000-01-15",
  "termsAccepted": true,
  "privacyPolicyAccepted": true
}
```

Expected: HTTP 200 and `"User registered successfully"`. `gender` accepts `MALE`, `FEMALE`, or `OTHERS`; `maritalStatus` accepts `SINGLE`, `MARRIED`, `DIVORCED`, or `WIDOWED`; `employmentType` accepts `STUDENT`, `SELF_EMPLOYED`, `EMPLOYED`, `UNEMPLOYED`, `OTHERS`, or `RETIRED`. The date must be in the past. To verify consent enforcement, send either consent field as `false` or omit it; expect HTTP 400. The email must have been OTP-verified and be inside the registration window.

### 4. Login

`POST {{baseUrl}}/auth/login`

```json
{
  "email": "{{testEmail}}",
  "password": "{{testPassword}}"
}
```

Expected: HTTP 200, a JSON `token`, and a `Set-Cookie` header for `refreshToken`. Add this Postman **Tests** script to store the access JWT:

```javascript
const body = pm.response.json();
pm.environment.set("accessToken", body.token);
```

### 5. Refresh access token

`POST {{baseUrl}}/auth/refresh`

No request body is needed. Send the `refreshToken` cookie set by login. Expected: HTTP 200, a new access token, and a rotated refresh cookie. Reuse of the old refresh cookie is rejected and revokes that refresh session. Update the Postman **Tests** script as for login so `accessToken` is replaced with the newly returned token.

### 6. Read and save medical history

Read existing state with `GET {{baseUrl}}/user/history/medical`, then save with:

`PUT {{baseUrl}}/user/history/medical`

Headers: `Authorization: Bearer {{accessToken}}`. Body type: raw JSON.

```json
{
  "details": "No known ongoing medical conditions. No current medications or allergies."
}
```

Use only appropriate dummy information during tests. If the user has nothing to report, `"details": "None known"` is valid. Expected: HTTP 200 and `{ "details": "...", "completed": true }`.

### 7. Read and save family history

Read existing state with `GET {{baseUrl}}/user/history/family`, then save with:

`PUT {{baseUrl}}/user/history/family`

Headers: `Authorization: Bearer {{accessToken}}`. Body type: raw JSON.

```json
{
  "details": "No known family history relevant to mental health."
}
```

`"details": "None known"` is valid when there is nothing to report. Expected: HTTP 200 and `{ "details": "...", "completed": true }`. Blank or more than 10,000 characters returns HTTP 400.

### 8. Start a session

`POST {{baseUrl}}/session/start`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 201 and an `IN_PROGRESS` session response. Add this **Tests** script to save its ID:

```javascript
const body = pm.response.json();
pm.environment.set("sessionId", body.id);
```

Starting another session before finishing or abandoning this one returns HTTP 409. If current legal consent is missing or out of date, start returns HTTP 403.

### 9. Find the unfinished session

`GET {{baseUrl}}/session/draft`

Headers: `Authorization: Bearer {{accessToken}}`.

Expected: HTTP 200 with the current unfinished session, or HTTP 204 with no body if none exists.

### 10. Get the active question bank

`GET {{baseUrl}}/questions/active`

Headers: `Authorization: Bearer {{accessToken}}`

Expected: HTTP 200 with the bank version and variants. Use the next-question route below to get the server-selected prompt for a session.

### 11. Get the next question

`GET {{baseUrl}}/session/{{sessionId}}/question/next`

Headers: `Authorization: Bearer {{accessToken}}`

Expected: HTTP 200 with the next question. Add this Tests script to save its ID for the turn request:

```javascript
const body = pm.response.json();
if (body.question) pm.environment.set("questionVariantId", body.question.id);
```

When every active theme has been answered, `conversationComplete` is `true` and `question` is null.

### 12. Submit a text response

`POST {{baseUrl}}/session/{{sessionId}}/turns`

Headers: `Authorization: Bearer {{accessToken}}`

```json
{
  "questionVariantId": {{questionVariantId}},
  "responseText": "I have been feeling low and have had trouble sleeping."
}
```

Expected: HTTP 201. Save a new question ID by calling the next-question endpoint before each following response. If the crisis flag is raised, the session is paused, guidance is returned, and a resource panel should be shown; inspect the response's `crisisResources` list. Generic guidance is returned even if no configured resources exist.

The backend sends JSON to the ML orchestrator over HTTP/1.1. This is intentional: the JDK HTTP client otherwise attempts an HTTP/2 cleartext (`h2c`) upgrade, which Uvicorn does not support and may leave FastAPI with an empty request body (422). If you run an older backend build, restart it after rebuilding so the HTTP/1.1 setting is applied. The WebSocket/upgrade warnings in Uvicorn are unrelated to Postman itself.

### 13. Read saved turns

`GET {{baseUrl}}/session/{{sessionId}}/turns`

Headers: `Authorization: Bearer {{accessToken}}`

Expected: HTTP 200 with turns in order. Only the owner can read these; stored response text is decrypted by the backend for this response.

### Audio response upload

For a Postman-ready dummy WAV file, run this from a terminal with Python 3 (it writes a short, silent, valid PCM WAV file in the current directory):

```powershell
python -c "import wave,struct; w=wave.open('dummy.wav','wb'); w.setparams((1,2,16000,1600,'NONE','not compressed')); w.writeframes(b''.join(struct.pack('<h',0) for _ in range(1600))); w.close()"
```

First call `GET {{baseUrl}}/session/{{sessionId}}/question/next` and store `question.id` as `questionVariantId`. Send:

`POST {{baseUrl}}/session/{{sessionId}}/turns/audio`

Headers: `Authorization: ******

Choose **Body → form-data** and add `questionVariantId` as Text and `file` as File; select `dummy.wav` for `file`. Do not manually set the multipart `Content-Type` header: Postman must add its boundary. Expected: HTTP 201 with audio metadata and `analysisStatus: "PENDING_MODEL_ANALYSIS"`. Repeat question retrieval and audio or text submission until each theme in the session's pinned bank has one answer.

### Generate and retrieve final assessment

After all themes are answered, call `POST {{baseUrl}}/session/complete/{{sessionId}}`, then `POST {{baseUrl}}/session/{{sessionId}}/assessment` with the access token and no body. When ML analysis is implemented and reachable, expected HTTP 201 response shape:

```json
{
  "id": 12,
  "sessionId": 34,
  "score": 0.62,
  "probability": 0.62,
  "severity": "MODERATE",
  "explanation": "Example only: generated by the configured model.",
  "analyses": [
    {"modality": "TEXT", "score": 0.58, "probability": 0.58},
    {"modality": "VOICE", "score": 0.66, "probability": 0.66},
    {"modality": "FUSION", "score": 0.62, "probability": 0.62}
  ],
  "completedAt": "2026-01-01T12:00:00"
}
```

This response is illustrative only, not a model score. The current text-analysis, voice-analysis, and fusion endpoints in the ML service are placeholders, so the real expected result today is HTTP 503 with no saved assessment. Once a result exists, `GET {{baseUrl}}/session/{{sessionId}}/assessment` returns the saved result without re-running inference. Assessment analysis can be retried after ML is available.

### 14. Get a session by ID

`GET {{baseUrl}}/session/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`.

Expected: HTTP 200 with the session. An ID that is absent or belongs to another user returns HTTP 404.

### 15. Resume a session

`POST {{baseUrl}}/session/resume/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with the session in `IN_PROGRESS`. The start endpoint creates this status, so this request can be tested immediately after starting. Only an owned unfinished session can be resumed; current legal consent is required.

### 16. Complete a session

`POST {{baseUrl}}/session/complete/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with status `COMPLETED` and a `completedAt` timestamp after one response was saved for every active theme. Otherwise HTTP 409 explains the question-completion prerequisite. A completed session cannot be resumed or abandoned.

### 17. Abandon a session

To test abandonment separately, create another session after completing the first one, then save its new ID to `sessionId` using the start script above.

`POST {{baseUrl}}/session/abandon/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with status `ABANDONED`. The record remains stored, but it is no longer an unfinished session.

### 18. Logout

`POST {{baseUrl}}/auth/logout`

No request body is needed. Send the current `refreshToken` cookie. Expected: HTTP 200 and an expired `refreshToken` cookie. The access token associated with that refresh session is invalid immediately. Verify by calling `GET {{baseUrl}}/session/draft` with the old `accessToken`; expect HTTP 401. Logout also succeeds and clears the cookie if no cookie is present.

## Typical error responses

Most application errors are plain-text messages. Validation errors are JSON maps from field name to error text.

| HTTP status | Examples |
|---|---|
| 400 | Malformed JSON, invalid request fields, invalid/expired OTP, missing/false consent, blank/oversized history details or text response. |
| 401 | Missing or invalid access JWT, invalid refresh token, invalid login credentials. |
| 403 | Email is not verified for registration; current legal consent is required for session start/resume. |
| 404 | Session does not exist or is owned by another user. |
| 409 | Account already exists; an unfinished session already exists; invalid lifecycle transition; medical/family history incomplete before starting; response submitted to a non-next question; session completed before all active question themes were answered. |
| 429 | OTP resend requested before the cooldown has elapsed. |
| 503 | ML analysis service is unavailable or returned an invalid result; no response turn is persisted. |

## Postman negative and error-case cookbook

Use the local Postman environment from **Postman quick start**. All data below is fabricated; do not use a real person's health history. For protected routes, set `Authorization` to `Bearer {{accessToken}}`. Do not send an access token as a query parameter or request-body field. Most handled application errors return plain text; bean-validation failures return a JSON object mapping field names to messages.

### Authentication errors

| Endpoint | Negative test setup / request | Expected result |
|---|---|---|
| `POST /auth/send-otp` | `{"email":"not-an-email"}` | `400 Bad Request`; validation JSON includes an email error. |
| `POST /auth/send-otp` | Resend to the same unverified email within 60 seconds. | `429 Too Many Requests`; wait 60 seconds before retrying. |
| `POST /auth/send-otp` | Request OTP for already registered `{{testEmail}}`. | `409 Conflict`, user already exists. |
| `POST /auth/verify-otp` | `{"email":"{{testEmail}}","otp":"12"}` | `400 Bad Request`; OTP must be six digits. |
| `POST /auth/verify-otp` | Send a six-digit value other than the email code. Repeat up to five times to burn the code. | `400 Bad Request`; after five failed attempts request a new OTP. Expired OTP also returns 400. |
| `POST /auth/register` | Submit a valid registration body for a fresh email without verifying an OTP for it. | `403 Forbidden`, `Email not verified`. |
| `POST /auth/register` | Use valid registration fields but set either consent field to `false`. | `400 Bad Request`; no account is created. |
| `POST /auth/register` | Register the same email again after successful registration. | `409 Conflict`, user already exists. |
| `POST /auth/login` | `{"email":"{{testEmail}}","password":"WrongPassword!2026"}` | `401 Unauthorized`, invalid credentials. |
| `POST /auth/login` | Omit `password` or send a blank password. | `400 Bad Request`; validation JSON identifies the password field. |
| `POST /auth/refresh` | Remove the `refreshToken` cookie in Postman's cookie manager, then send the request. | `401 Unauthorized`, invalid refresh token. |
| `POST /auth/refresh` | Save the old refresh cookie before a successful refresh, then resend that old cookie after rotation. | `401 Unauthorized`; replay revokes the refresh session, so log in again. |
| `POST /auth/logout` | Send with no refresh cookie. | `200 OK`; logout is idempotent and expires the cookie. |

Use this valid-shaped registration body with a unique, unverified email to test the verification gate:

```json
{
  "email": "psyche.unverified@example.com",
  "password": "LocalTestPass!2026",
  "firstName": "Test",
  "lastName": "User",
  "gender": "OTHERS",
  "maritalStatus": "SINGLE",
  "jobTitle": "Student",
  "employmentType": "STUDENT",
  "phoneNumber": "+919876543210",
  "workingHoursPerWeek": 10,
  "dateOfBirth": "2000-01-15",
  "termsAccepted": true,
  "privacyPolicyAccepted": true
}
```

To test consent validation, verify a different fresh email, then send the regular registration body with `"termsAccepted": false` (or `"privacyPolicyAccepted": false`). Expect HTTP 400. A fixed dummy OTP does not exist: use the code delivered to the configured test inbox. OTP codes expire after five minutes, and successful verification allows registration for 15 minutes.

### Protected-route and history errors

| Endpoint | Negative test setup / request | Expected result |
|---|---|---|
| Any protected endpoint | Remove `Authorization` or use `Bearer invalid-token`. | `401 Unauthorized`. |
| `GET /user/history/medical` and `GET /user/history/family` | Call before saving the corresponding history. | `200 OK` with `{"details":null,"completed":false}`; incomplete state is not itself an error. |
| `PUT /user/history/medical` and `PUT /user/history/family` | `{"details":"   "}` | `400 Bad Request`; details must not be blank. |
| Same history `PUT` routes | Omit `details`, send `{"details":null}`, or submit more than 10,000 characters. | `400 Bad Request`. |

Valid dummy history for either `PUT` route is `{"details":"None known (Postman test data)."}`. The API has no client-supplied user ID: each authenticated user can only read or replace their own history.

### Session and question-flow errors

| Endpoint | Negative test setup / request | Expected result |
|---|---|---|
| `POST /session/start` | Log in as a newly registered test user and start before saving both history forms. | `409 Conflict`: both medical and family histories must be complete. Save both using the history `PUT` requests, then start succeeds. |
| `POST /session/start` | Start once, then start again without completing or abandoning the first session. | `409 Conflict`; resume or abandon the existing session first. |
| `POST /session/start` or `/session/resume/{sessionId}` | Change server `TERMS_VERSION` or `PRIVACY_POLICY_VERSION` after acceptance, restart, and try the request. Restore the original values after testing. | `403 Forbidden`; the user must accept the currently configured legal-document versions. There is no consent-renewal endpoint yet. |
| `GET /session/draft` | Call with no unfinished session. | `204 No Content` with an empty body. |
| `GET /session/{sessionId}` | Use nonexistent numeric ID `99999999` or an ID owned by a different test account. | `404 Not Found`; the API does not reveal whether another user's session exists. |
| `POST /session/resume/{sessionId}` | Use a nonexistent/other user's ID. | `404 Not Found`. |
| `POST /session/resume/{sessionId}` | Try to resume a `COMPLETED` or `ABANDONED` session. | `409 Conflict`; only unfinished sessions can be resumed. |
| `POST /session/complete/{sessionId}` | Complete a new session before answering any questions. | `409 Conflict`; answer one question from every active theme first. |
| `POST /session/complete/{sessionId}` | Call again after completion. | `409 Conflict`; only in-progress or paused sessions can be completed. |
| `POST /session/abandon/{sessionId}` | Abandon a completed/already abandoned session. | `409 Conflict`; only unfinished sessions can be abandoned. |
| `GET /questions/active` | Omit the access token. | `401 Unauthorized`. |
| `GET /session/{sessionId}/question/next` | Use a nonexistent/other user's session ID. | `404 Not Found`. |
| `GET /session/{sessionId}/question/next` | Use a completed/abandoned session. | `409 Conflict`. For a `PAUSED` session expect HTTP 200 with `available:false`, not an error. |

To test the missing-history gate, use a brand-new test account before either history `PUT`; to test the duplicate-session error, start once and repeat `/session/start` without first completing or abandoning. No API is provided to directly toggle legal-consent state, so the stale-consent test requires changing the configured version and restarting the backend.

### Turn submission errors and crisis path

For `POST /session/{{sessionId}}/turns`, first call `GET /session/{{sessionId}}/question/next` and save `question.id` as `questionVariantId`. The question ID must be the one currently returned for that session.

| Negative test request | Expected result |
|---|---|
| `{"questionVariantId":null,"responseText":"Test answer"}` | `400 Bad Request`; question ID is required. |
| `{"questionVariantId":{{questionVariantId}},"responseText":"   "}` | `400 Bad Request`; response text cannot be blank. |
| Use an ID other than the currently returned `questionVariantId`. | `409 Conflict`; retrieve the current next prompt and use its ID. |
| Use a valid next prompt with a completed, paused, or abandoned session. | `409 Conflict`; a response can only be added to an in-progress session. Resume a paused session first. |
| Stop the ML engine or configure `ML_ENGINE_BASE_URL` to an unavailable address, then submit an otherwise valid turn. | `503 Service Unavailable`; the response is not persisted. Restore the ML service/URL afterward. |
| `GET /session/{{sessionId}}/turns` without a token / with another user's session ID. | `401 Unauthorized` / `404 Not Found`, respectively. |

### Audio and final-assessment errors

| Negative test request | Expected result |
|---|---|
| Upload without the `file` part or with an empty file. | `400 Bad Request` (or a multipart binding error). |
| Upload a `.txt` file, a non-audio content type, or bytes that do not match the declared audio signature. | `400 Bad Request`; no turn should be saved. |
| Upload a valid WAV but use a different question ID from the next-question response. | `409 Conflict`; no audio turn is saved. |
| Upload audio to another user's session or a completed/paused session. | `404 Not Found` / `409 Conflict`, respectively. |
| Call `POST /session/{{sessionId}}/assessment` before completion. | `409 Conflict`; complete the session first. |
| Call `POST /session/{{sessionId}}/assessment` after completion while the ML text/voice/fusion route is unavailable or still a placeholder. | `503 Service Unavailable`; no partial assessment result is saved. Retry after the ML service implements the required routes. |
| Call `GET /session/{{sessionId}}/assessment` before a result has been generated, or use another user's session ID. | `404 Not Found`. |

This example deliberately submits the wrong question ID and should return HTTP 409:

```json
{
  "questionVariantId": 99999999,
  "responseText": "This ID is deliberately not the next question."
}
```

An oversized response (more than 10,000 characters) returns HTTP 400. To exercise the crisis-handling path (not an error), submit an otherwise valid response such as `I am thinking about suicide and may hurt myself`. If the ML keyword matcher detects it, expect HTTP 201, `crisisDetected:true`, non-diagnostic guidance, configured crisis resources (including India's Tele-MANAS contact), and a paused session. Detection is heuristic and not guaranteed; this is not a safety assessment. For a normal positive-path response use `I have been feeling okay and sleeping well this week.`

### Example error bodies

Validation failures return a JSON field map (exact wording depends on the invalid field):

```json
{
  "responseText": "must not be blank"
}
```

Handled application errors generally return plain text, for example:

```text
Complete both medical history and family history before starting an assessment session.
```

```text
Turn analysis is temporarily unavailable. Your response was not saved; please try again.
```

In Postman, assert the HTTP status and relevant field/message rather than byte-for-byte error text. Audio uploads use multipart form-data; assessment generation has no request body. Until the ML model routes are implemented, the documented 503 response is expected for final assessment generation.

## Running and testing

From the repository root on Windows:

```powershell
.\backend\mvnw.cmd -f backend\pom.xml test
```

Run the API from the `backend` directory with the Maven wrapper after providing the required environment variables:

```powershell
.\mvnw.cmd spring-boot:run
```

## Security and deployment notes

- Use HTTPS in production and keep the refresh cookie `Secure`, `HttpOnly`, and `SameSite=Strict`.
- Restrict credentialed CORS to the exact trusted frontend origins.
- Keep `JWT_SECRET`, `ENCRYPTION_KEY`, database credentials, and mail credentials out of source control and exported Postman collections.
- Refresh cookies are browser credentials; protect the `/auth/refresh` and `/auth/logout` routes with deployment-level CSRF/origin controls appropriate to the deployed frontend topology. `SameSite=Strict` and CORS are useful controls, not a substitute for reviewing the complete production deployment.
- Access JWT revocation incurs a database lookup per protected request; monitor its latency and database health.
