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
| `GET` | `/user/history/medical` | Access JWT | Get the authenticated user's medical-history details and completion state. |
| `PUT` | `/user/history/medical` | Access JWT | Create or update the authenticated user's medical history. |
| `GET` | `/user/history/family` | Access JWT | Get the authenticated user's family-history details and completion state. |
| `PUT` | `/user/history/family` | Access JWT | Create or update the authenticated user's family history. |

No turn submission, pause, question selection, crisis check, assessment result, or history-list endpoints are currently implemented. Model classes in the repository do not imply that an HTTP endpoint exists.

## Requirements and configuration

- Java 27 and the included Maven wrapper (`mvnw.cmd` on Windows).
- MySQL database, configured with `DB_URL`, `MYSQLUSER`, and `MYSQLPASSWORD`.
- Email delivery configured with `MAIL_USERNAME` and `MAIL_PASSWORD`; send-OTP requires a functioning mail provider.
- `JWT_SECRET` must be Base64-encoded key material of at least 256 bits. Provide it through the environment; do not commit it.
- Configure `CORS_ALLOWED_ORIGINS` as a comma-separated list of trusted browser frontend origins.
- `COOKIE_SECURE` defaults to `true`; use HTTPS and leave it enabled in production. For local HTTP-only development, it may be set to `false`.
- `TERMS_VERSION` and `PRIVACY_POLICY_VERSION` default to `v1`.

The application uses `spring.jpa.hibernate.ddl-auto=update` to update the configured database schema. Back up and review production database changes before deploying schema updates.

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

The frontend keeps access tokens in JavaScript memory only; they are not written to `localStorage` or `sessionStorage` and are cleared on logout. A full page reload obtains a fresh access token through the `HttpOnly` refresh cookie. Protected API calls also attempt one refresh when the access token is missing or receives HTTP 401, then retry the original request once. Concurrent refresh attempts within the same page share one in-flight refresh request. Postman can retain the refresh cookie in its cookie jar; do not paste real credentials or token values into shared collections or source control.

Memory-only access-token storage reduces exposure to persistent token theft from browser storage, but it does not prevent an active same-origin XSS from issuing authenticated requests while a user is signed in. Continue to avoid unsafe HTML/script injection and keep frontend dependencies maintained.

## Registration consent

Account creation requires both the current Terms of Service and Privacy Policy to be accepted. The server rejects either value missing or `false`; successful registration stores each document version and acceptance timestamp. The current values are configured via `TERMS_VERSION` and `PRIVACY_POLICY_VERSION`. Updating a document requires deploying its new version value. Existing users are not automatically considered to have accepted a new version, so starting or resuming a session is blocked if their stored versions are outdated. An endpoint for existing users to renew consent is not implemented yet.

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

Starting creates an `IN_PROGRESS` session immediately. Only one unfinished session (`DRAFT`, `IN_PROGRESS`, or `PAUSED`) is allowed per user. A second start returns HTTP 409 until that unfinished session is resumed or abandoned. Starting and resuming require acceptance timestamps and versions matching the currently configured Terms and Privacy Policy. Starting additionally requires the medical and family history completion flags. Completing or abandoning does not require current consent or profile completion, so the user can close an existing session.

Allowed lifecycle behavior:

| Operation | Accepted state | Result |
|---|---|---|
| Start | No unfinished session | Creates `IN_PROGRESS` (HTTP 201). |
| Find draft | Any | Returns the newest unfinished session (HTTP 200), or no body (HTTP 204). |
| Get session | Any existing owned session | Returns session data (HTTP 200). |
| Resume | `DRAFT`, `PAUSED`, or `IN_PROGRESS` | Sets/resets activity and returns `IN_PROGRESS` (HTTP 200). `IN_PROGRESS` is effectively idempotent. |
| Complete | `IN_PROGRESS` or `PAUSED` | Sets `COMPLETED` and `completedAt` (HTTP 200). |
| Abandon | `DRAFT`, `IN_PROGRESS`, or `PAUSED` | Sets `ABANDONED` and retains the record (HTTP 200). |

There is currently no pause operation or API which creates a `DRAFT` session. The resume endpoint can nevertheless be exercised against an `IN_PROGRESS` session created by `/session/start`.

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

### 10. Get a session by ID

`GET {{baseUrl}}/session/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`.

Expected: HTTP 200 with the session. An ID that is absent or belongs to another user returns HTTP 404.

### 11. Resume a session

`POST {{baseUrl}}/session/resume/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with the session in `IN_PROGRESS`. The start endpoint creates this status, so this request can be tested immediately after starting. Only an owned unfinished session can be resumed; current legal consent is required.

### 12. Complete a session

`POST {{baseUrl}}/session/complete/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with status `COMPLETED` and a `completedAt` timestamp. A completed session cannot be resumed or abandoned.

### 13. Abandon a session

To test abandonment separately, create another session after completing the first one, then save its new ID to `sessionId` using the start script above.

`POST {{baseUrl}}/session/abandon/{{sessionId}}`

Headers: `Authorization: Bearer {{accessToken}}`. No request body.

Expected: HTTP 200 with status `ABANDONED`. The record remains stored, but it is no longer an unfinished session.

### 14. Logout

`POST {{baseUrl}}/auth/logout`

No request body is needed. Send the current `refreshToken` cookie. Expected: HTTP 200 and an expired `refreshToken` cookie. The access token associated with that refresh session is invalid immediately. Verify by calling `GET {{baseUrl}}/session/draft` with the old `accessToken`; expect HTTP 401. Logout also succeeds and clears the cookie if no cookie is present.

## Typical error responses

Most application errors are plain-text messages. Validation errors are JSON maps from field name to error text.

| HTTP status | Examples |
|---|---|
| 400 | Malformed JSON, invalid request fields, invalid/expired OTP, missing/false consent, blank/oversized history details. |
| 401 | Missing or invalid access JWT, invalid refresh token, invalid login credentials. |
| 403 | Email is not verified for registration; current legal consent is required for session start/resume. |
| 404 | Session does not exist or is owned by another user. |
| 409 | Account already exists; an unfinished session already exists; invalid lifecycle transition; medical/family history incomplete before starting. |
| 429 | OTP resend requested before the cooldown has elapsed. |

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
- Keep `JWT_SECRET`, database credentials, and mail credentials out of source control and exported Postman collections.
- Refresh cookies are browser credentials; protect the `/auth/refresh` and `/auth/logout` routes with deployment-level CSRF/origin controls appropriate to the deployed frontend topology. `SameSite=Strict` and CORS are useful controls, not a substitute for reviewing the complete production deployment.
- Access JWT revocation incurs a database lookup per protected request; monitor its latency and database health.
