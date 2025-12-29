### AI Study Buddy — Product Requirements Document (PRD)

### Document control
- **Product**: AI Study Buddy
- **Version**: 1.0
- **Last updated**: 2025-12-29
- **Status**: Draft (Capstone prototype scope)

### Summary
AI Study Buddy is a full‑stack learning platform that combines:
- **A web app** (React + Vite) for authentication, course/chapter organization, AI chat, device pairing, RAG-based tutoring, and focus dashboards.
- **A Firebase backend** (Auth, Firestore, Storage, Cloud Functions) integrated with **Genkit + Gemini** for conversational AI and device APIs.
- **A Raspberry Pi focus tracker** running an on‑device TensorFlow Lite model and temporal logic, uploading **only aggregated focus-session summaries** (no raw video).

### Problem statement
Students often struggle to sustain focus and to manage study time in a data‑informed way. Existing planners/timers/chatbots do not objectively measure engagement, and cloud video monitoring is invasive. AI Study Buddy pairs AI tutoring with privacy‑preserving, edge‑computed focus metrics to help students reflect on both **what** they studied and **how** they studied.

### Goals
- **G1**: Students can authenticate and manage **courses, course sessions (e.g., chapters), and AI chats**.
- **G2**: Students can chat with an **AI tutor** in chats that belong to a course session, with persisted history and context.
- **G3**: Students can **pair a Raspberry Pi** to their account via a claim code.
- **G4**: Students can start/stop **focus tracking sessions** (separate from courses/sessions/chats); Raspberry Pi detects focus-session start/stop via backend coordination, runs inference locally, and uploads **aggregated** focus summaries.
- **G5**: Web app displays focus-session metrics on a **focus dashboard**.

### Non-goals (prototype scope boundaries)
- Production-grade scale, cost optimization, and advanced analytics across many users
- Mobile/native apps
- Long-term controlled user studies
- Always-on video streaming, cloud storage of frames, or per-frame logging in cloud
- Sophisticated model personalization or multi-user device sharing (unless explicitly added)

### Target users & personas
- **Student (primary)**: wants course/chapter organization, AI help grounded in their materials, and objective focus feedback.
- **Project team / developer (secondary)**: needs debugging visibility and reliable integration for demos.

### User journeys (happy paths)
- **J1: Study with AI**
  - Sign in → create/select course → create/select course session (e.g., “Chapter 3”) → create/select AI chat → send message → receive streamed AI response → messages persist.
- **J2: Pair device**
  - Open “Pair device” → enter claim code → device becomes associated with user → device receives token and becomes “paired”.
- **J3: Focus tracking (separate feature)**
  - While studying (inside a course session), student optionally starts focus tracking (select device) → Pi detects active focus session → logs focus state → student stops focus tracking (or study ends) → Pi uploads summary → dashboard shows metrics.

### User stories
- **US1**: As a student, I can sign up/sign in/sign out and stay signed in across refresh.
- **US2**: As a student, I can create and manage courses and course sessions (e.g., chapters).
- **US3**: As a student, I can create multiple AI chats inside a course session and see my message history.
- **US4**: As a student, I can message an AI tutor and receive responses quickly (streaming).
- **US5**: As a student, I can upload study materials to a course and view/download them later.
- **US5a**: As a student, I want the AI to use my uploaded course materials (RAG) so answers are grounded in what I’m studying.
- **US6**: As a student, I can pair my Raspberry Pi using a claim code and see that it’s paired.
- **US7**: As a student, I can choose whether to start focus tracking (opt-in) and can start/stop it at any time during a study session.
- **US8**: As a student, I can trust that no raw video is uploaded or stored.

---

### Functional requirements

### Web application (React + Vite)

### Authentication
- **FR-W1**: Support sign up, sign in, sign out using Firebase Auth.
- **FR-W2**: Persist auth state; protect authenticated routes; handle loading/error states.

### Course/session/chat management
- **FR-W3**: CRUD **Courses** scoped to the authenticated user.
- **FR-W4**: CRUD **Course Sessions** (e.g., chapters) linked to a course and scoped to the user.
- **FR-W5**: CRUD **Chats** linked to a course session.
- **FR-W6**: Create/read **Messages** linked to a chat; messages update in real time (Firestore listeners).
- **FR-W7**: Navigation UX to select course → session → chat (sidebar or equivalent).

### AI chat UX
- **FR-W8**: Create a new chat thread via backend (`/createChat`) and persist metadata.
- **FR-W9**: Send user messages and receive AI responses via backend (`/chat`).
- **FR-W10**: Support streamed response rendering (SSE) with partial tokens.
- **FR-W11**: Persist user + assistant messages to Firestore with timestamps and sender role.
- **FR-W12**: Robust error handling (auth failures, SSE disconnects, timeouts, invalid payloads).

### File upload
- **FR-W13**: Upload file(s) to Firebase Storage and store metadata in Firestore.
- **FR-W14**: Enforce basic client-side validation (type/size), progress UI, and retry/error states.
- **FR-W15**: Allow viewing/downloading uploaded files from the course context.

### RAG (course materials grounding)
- **FR-W15a**: Allow enabling/disabling “Use course materials” per chat or per message (UX decision).
- **FR-W15b**: Display AI citations/snippets (file name + excerpt) when the AI uses course materials (if supported by backend).

### Device pairing UX
- **FR-W16**: Provide a guided “Pair Raspberry Pi” flow using a short claim code.
- **FR-W17**: Claim device and associate it with the current user in Firestore.
- **FR-W18**: Show pairing status (pending/paired/failed) and basic device info (deviceId, last seen if available).

### Focus session control (backend coordination with device)
- **FR-W19**: Before a student begins studying in a course session, prompt them to optionally enable focus tracking (opt-in).
- **FR-W19a**: While studying, allow starting/stopping focus tracking at any time (explicit UI control).
- **FR-W19b**: Focus tracking must be off by default; the student can proceed without it.
- **FR-W19c**: When starting focus tracking, allow selecting a paired device (or default to the last-used device).
- **FR-W19d**: When stopping focus tracking, clearly confirm that a summary will be generated/uploaded.
- **FR-W20**: Persist active focus session state such that the Pi can query it via `/device/currentFocusSession`.
- **FR-W21**: Handle edge cases (attempting to start multiple active focus sessions; device not paired; stale device).

### Focus dashboard
- **FR-W22**: Fetch/subscribe to focus summaries for the selected **focus session**.
- **FR-W23**: Display metrics (see “Focus summary schema” below).
- **FR-W24**: Handle “no summary yet” and “summary pending upload” states gracefully.
- **FR-W25**: If focus tracking was not enabled for a study session, show a friendly “Focus tracking was not started” empty-state (not an error).
- **FR-W26**: Provide a global “Focus Analytics / History” view listing past focus sessions and their summaries.
- **FR-W27**: In global analytics, allow filtering/grouping by:
  - **All focus sessions** (including unlinked)
  - **Course** (courseId)
  - **Course session / chapter** (courseSessionId)

---

### Backend (Firebase Cloud Functions + Firestore + Genkit/Gemini)

### AI endpoints
- **FR-B1**: `POST /createChat`
  - Requires Firebase Auth user.
  - Creates Genkit session and corresponding Firestore chat record.
- **FR-B2**: `POST /chat` (SSE streaming)
  - Requires Firebase Auth user.
  - Loads Genkit session; generates response via Gemini; streams tokens.
  - Writes user message and assistant message to Firestore; updates Genkit session store.
- **FR-B3**: `GET/POST /getSessionHistory` (debug)
  - Access-controlled (prefer admin/developer-only).
  - Returns stored Genkit session history/state.

### Course materials + RAG endpoints (high-level; exact design TBD)
- **FR-B3a**: Ingest course files for retrieval (extract text, chunk, embed, store).
- **FR-B3b**: At chat time, retrieve relevant chunks for the user’s course and provide them as grounded context to the model.
- **FR-B3c**: Optionally return citations (file + chunk excerpt) alongside AI responses.

### Device endpoints
- **FR-B4**: `POST /device/register`
  - Device registers a claim code and enters “awaiting claim”.
- **FR-B5**: `GET /device/pairingStatus`
  - Device polls pairing completion; once claimed returns a **device token**.
- **FR-B6**: `GET /device/currentFocusSession`
  - Auth: device token.
  - Returns current active focus session assignment for that device (focusSessionId or none).
- **FR-B7**: `POST /device/sessionSummary`
  - Auth: device token.
  - Validates/normalizes payload and stores summary in Firestore under the correct user+focusSession.
  - Prevents spoofing: ensure device belongs to the user/session association.
  - **Note**: Focus tracking is user-controlled; backend must not start focus sessions without an explicit user action recorded by the web app.

### Data integrity & validation
- **FR-B8**: Validate required fields, types, ranges, and timestamps for all endpoints.
- **FR-B9**: Ensure idempotency where practical (e.g., avoid duplicate summaries for same session unless explicitly allowed).

---

### Raspberry Pi agent (edge focus tracker)

### Pairing
- **FR-P1**: Generate/display claim code; call `/device/register`.
- **FR-P2**: Poll `/device/pairingStatus` until paired; securely persist device token locally.
- **FR-P3**: On reboot, reuse stored token; re-pair only if token invalid/expired.

### Session coordination
- **FR-P4**: Poll `/device/currentFocusSession` using device token (outbound HTTPS only).
- **FR-P4a**: Polling should be configurable and efficient (recommended: long-poll “wait up to N seconds for change” or short polling with exponential backoff while idle).
- **FR-P5**: Detect transition:
  - none → active focus session: start capture/inference/logging.
  - active focus session → none: stop and summarize.

### Capture + inference
- **FR-P6**: Capture frames locally (OpenCV).
- **FR-P7**: Preprocess frames to model input shape.
- **FR-P8**: Run inference with TensorFlow Lite runtime and output focused vs not-focused prediction.

### Temporal focus logic (state machine)
- **FR-P9**: Implement temporal smoothing:
  - Mark **DISTRACTED** only after ~30s continuous not-focused.
  - Mark **FOCUSED** after ~1–3s continuous focused while distracted.
  - **Note**: choose and document the exact refocus threshold for consistency across code + report.
- **FR-P10**: Log state transitions with timestamps for summary computation (local only).

### Summary + upload
- **FR-P11**: Compute focus summary JSON (schema below).
- **FR-P12**: Store summary locally for reliability.
- **FR-P13**: Upload summary to `/device/sessionSummary` with retry/backoff; queue if offline (push on stop; no polling required for uploads).

### Service robustness
- **FR-P14**: Run as a system service (e.g., systemd) with auto-restart on crash.
- **FR-P15**: Avoid storing or transmitting raw frames; keep sensitive data in-memory only.

---

### Focus summary schema (focus-session-level, aggregated)
Stored in Firestore and displayed in the dashboard:
- `userId` (implicit via path / ownership)
- `focusSessionId` (string)
- `deviceId` (string)
- `courseId` (string, optional; present when focus tracking was started from within a course)
- `courseSessionId` (string, optional; present when focus tracking was started from within a course session/chapter)
- `startTs` (timestamp)
- `endTs` (timestamp)
- `focusedMs` (number)
- `distractedMs` (number)
- `longestFocusedMs` (number)
- `longestDistractedMs` (number, optional if not computed)
- `distractions` (number; count of transitions into distracted)
- `avgFocusBeforeDistractMs` (number, optional)
- `focusPercent` (number 0–100)
- `createdAt` (timestamp, server time)

---

### Data model (Firestore) — conceptual
Exact paths may vary; must remain consistent across web/backend/device.
- `users/{userId}` (optional profile)
- `courses/{courseId}` with `userId`
- `courseSessions/{courseSessionId}` with `userId`, `courseId`
- `chats/{chatId}` with `userId`, `courseSessionId`, `genkitSessionId`
- `messages/{messageId}` with `chatId`, `userId`, `role`, `content`, `createdAt`
- `courseFiles/{fileId}` with `userId`, `courseId`, `storagePath`, metadata
- `ragChunks/{chunkId}` with `userId`, `courseId`, `fileId`, `text`, `embedding` (or reference), metadata
- `devices/{deviceId}` with `claimCode`, `status`, `pairedUserId`, `deviceTokenHash`, `lastSeenAt`
- `focusSessions/{focusSessionId}` with:
  - `userId`, `deviceId`, `startTs`, `endTs`, `status`
  - `courseId` (optional)
  - `courseSessionId` (optional)
- `focusSummaries/{summaryId}` (or nested under user/focusSession) with fields above
- `genkit_sessions/{genkitSessionId}` for Genkit persistence (backend-owned)

### API contracts (high-level)
- **Auth (web)**: Firebase ID token
- **Device auth**: device token issued at pairing; sent in an `Authorization` header (exact format defined by implementation)
- **Transport**: HTTPS only; AI chat uses SSE for streaming

---

### Non-functional requirements

### Privacy
- **NFR-PR1**: No raw video or frames uploaded or stored in cloud.
- **NFR-PR2**: Upload only aggregated focus session summaries and minimal device metadata.
- **NFR-PR3**: Focus tracking must be opt-in (explicit user choice); provide clear UI copy about what is and isn’t collected.

### Security
- **NFR-S1**: Enforce AuthN/AuthZ for all endpoints and Firestore/Storage rules.
- **NFR-S2**: Device token must be unguessable, stored securely, and validated server-side.
- **NFR-S3**: Users can read/write only their own Firestore data; device writes only permitted summaries for its paired user.

### Performance
- **NFR-P1**: AI chat feels interactive; streaming begins quickly (target defined by team).
- **NFR-P2**: Pi inference loop sustains a stable rate sufficient for temporal thresholds (e.g., ~10–15 FPS target).

### Reliability
- **NFR-R1**: Pi queues summaries offline and retries until successful.
- **NFR-R2**: Graceful degradation on function errors/timeouts; clear UI states.

### Usability
- **NFR-U1**: Pairing flow must be simple (enter claim code + confirm success).
- **NFR-U2**: Dashboard metrics must be interpretable to non-technical users.

---

### Success metrics (prototype)
- **SM1**: End-to-end demo works: login → chat → start focus session → stop focus session → summary visible on dashboard.
- **SM2**: Chat responses stream and persist reliably.
- **SM3**: Pi generates and uploads summaries without raw video leaving device.
- **SM4**: Focus dashboard shows correct metrics for the correct user/session.

---

### Milestones (suggested)
- **M1**: Stable data model + Firestore/Storage rules.
- **M2**: AI chat endpoints + web streaming UI productionized for demo reliability.
- **M3**: Pairing flow end-to-end (web + backend + Pi token).
- **M4**: Current-focus-session coordination end-to-end.
- **M5**: Summary ingestion + dashboard final metrics.
- **M6**: Replace inference stub with trained TFLite model; validate performance.

---

### Risks & mitigations
- **Model accuracy variability**: start with simple models; iterate dataset; consider quantization; communicate limitations.
- **Pi performance/stability**: reduce resolution/model size; implement watchdog + restart; tune capture loop.
- **Integration mismatches**: define schema/API early; add validation + versioning fields if needed.
- **Privacy/security misconfig**: strict rules, minimal data storage, token hashing and rotation strategy.


