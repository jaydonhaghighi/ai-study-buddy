# Feature #4 Implementation Tracker

## Feature
**Turn any chat/notes into quiz + flashcards**

## Goal
Let a student generate active-recall assets from a selected chat context:
- quick quiz (`MCQ + short answer`)
- flashcards (`spaced-repetition ready`)
- exam-style questions with rubric

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Definition of Done
- From a selected chat, user can click one action and generate a study set in under 20 seconds for normal-sized chats.
- A generated study set includes all 3 artifacts: quiz, flashcards, exam-style questions.
- Each generated item includes source grounding (`chat message refs and/or material citations`) for trust.
- User can interact with quiz/flashcards in-app and save review state.
- Flashcards store scheduling fields needed for spaced repetition (`nextReviewAt`, `intervalDays`, `easeFactor`, `repetitions`).
- Owner-only access is enforced for all new Firestore docs and function routes.

---

## Step 1: Scope + UX Contract
- [ ] Confirm v1 source scope: selected chat messages + indexed material chunks tied to that chat.
- [ ] Confirm v1 generation limits:
  - [ ] Quiz: 8-12 questions (mix of MCQ and short answer)
  - [ ] Flashcards: 12-20 cards
  - [ ] Exam questions: 3-5 with rubric
- [ ] Confirm user flow:
  - [ ] `Generate Study Set` button in right panel
  - [ ] loading/progress state
  - [ ] final tabs: `Quiz | Flashcards | Exam`
- [ ] Confirm failure UX (empty chat, no material, generation timeout, invalid response).

## Step 2: Data Model + Types
- [x] Add frontend types in `src/types.ts`:
  - [x] `StudySet`
  - [x] `QuizQuestion` (`mcq|short`, options, answer, explanation, sources)
  - [x] `Flashcard` (`front`, `back`, `tags`, `difficulty`, spaced-repetition fields)
  - [x] `ExamQuestion` (`prompt`, `rubric[]`, `modelAnswer`, `sources`)
- [~] Add Firestore collections:
  - [x] `studySets` (metadata + ownership + context refs)
  - [ ] `studySetItems` or subcollections (`quiz`, `flashcards`, `exam`) to avoid 1MB doc risk
  - [x] `flashcardReviews` (per-user progress events)
- [x] Add indexing strategy (likely `userId + chatId + createdAt desc`) in `firestore.indexes.json`.

## Step 3: Security Rules
- [~] Update `firestore.rules` for:
  - [x] `studySets`
  - [ ] `studySetItems`/subcollections
  - [x] `flashcardReviews`
- [ ] Keep owner-scoped create/read/update/delete pattern consistent with existing collections.
- [ ] Add emulator rule tests for unauthorized read/write attempts.

## Step 4: Backend Generation Endpoint (MVP)
- [x] Add `POST /studyGenerate` in `functions/src/index.ts`.
- [x] Request body:
  - [x] `userId`, `chatId`
  - [x] optional counts (`quizCount`, `flashcardCount`, `examCount`)
- [x] Validate chat ownership before any generation work.
- [x] Gather source context:
  - [x] recent chat messages from `messages` where `sessionId == chatId`
  - [x] ranked citations from `material_chunks` via existing retrieval helper pattern
- [x] Build one structured generation prompt that returns JSON only.
- [x] Validate/parses JSON response; reject malformed output with actionable error.
- [x] Persist generated set + items in Firestore with timestamps and status.
- [x] Return created `studySetId` and normalized payload for immediate UI render.

## Step 5: Backend Review Update Endpoint (Spaced Repetition)
- [x] Add `POST /flashcardReview` endpoint.
- [x] Input: `userId`, `studySetId`, `cardId`, `rating` (`again|hard|good|easy`), timestamp.
- [x] Compute next interval/ease-factor update (SM-2 style simplified rules).
- [x] Save updated scheduling fields and append review event.

## Step 6: Frontend Services + Hooks
- [~] Create `src/services/study-set-service.ts`:
  - [x] `generateStudySet(...)`
  - [x] `submitFlashcardReview(...)`
  - [ ] `loadStudySetsForChat(...)`
- [x] Create `src/components/chat/useStudySets.ts` hook for subscriptions and local state.
- [x] Wire into `Chat.tsx` near existing material/focus hooks.

## Step 7: UI Integration in Chat Experience
- [x] Add an `Active Recall` section in right sidebar (`ChatPreviewSidebar` + `ChatMaterialsPanel` adjacent section).
- [x] Add `Generate Study Set` button with count presets (`Quick`, `Standard`, `Exam Prep`).
- [x] Build study set viewer component:
  - [x] `Quiz` tab: answer inputs, reveal answers, score summary
  - [x] `Flashcards` tab: flip card UI + rating buttons for scheduling
  - [x] `Exam` tab: long-form prompts + rubric criteria display
- [x] Show source chips/citations on each generated item.

## Step 8: Prompt + Quality Hardening
- [~] Add explicit prompt rules:
  - [x] no invented facts
  - [x] include only source-grounded claims
  - [x] concise, student-friendly wording
  - [x] difficulty spread (easy/medium/hard)
- [~] Add server-side quality checks:
  - [x] minimum counts met
  - [x] non-empty answers/rubrics
  - [x] source references present
- [ ] Add fallback behavior when material context is sparse (use chat-only and label confidence).

## Step 9: Observability + Cost Controls
- [ ] Log generation latency, token usage (if available), parse failures, and retry counts.
- [ ] Enforce max chat-context window (message cap + truncation) to prevent runaway token costs.
- [ ] Add basic rate limiting per user (e.g., cooldown between generations).

## Step 10: QA Matrix
- [ ] Empty chat (no messages) -> clean validation error.
- [ ] Chat-only context (no uploaded materials) -> generates valid assets.
- [ ] Material-heavy chat -> source chips show file + location.
- [ ] Very long chat -> truncation still returns coherent set.
- [ ] Malformed model JSON -> server catches + user-friendly retry prompt.
- [ ] Unauthorized user hitting endpoint with foreign `chatId` -> 403.
- [ ] Flashcard review updates schedule fields correctly across ratings.
- [ ] Mobile layout check for quiz/flashcard interactions.

## Step 11: Rollout Plan
- [ ] Release behind feature flag (`studySetGenerationEnabled`).
- [ ] Internal dogfood on 5-10 real study sessions.
- [ ] Review generation quality + latency before broad enablement.
- [ ] Enable for all users and monitor errors for first 48 hours.

---

## Suggested Build Order (Execution Sequence)
1. Step 1 (scope contract)
2. Step 2 + Step 3 (data model + security)
3. Step 4 (generation endpoint)
4. Step 6 (frontend services/hooks)
5. Step 7 (UI integration)
6. Step 5 (flashcard review endpoint)
7. Step 8 + Step 9 (quality + observability)
8. Step 10 + Step 11 (QA + rollout)

## Active Blockers
- [ ] None currently.

## Change Log
- 2026-03-15: Initial Feature #4 execution plan created.
- 2026-03-15: Implemented MVP vertical slice (generation endpoint, flashcard review endpoint, Firestore rules/indexes, Active Recall sidebar UI with quiz/flashcards/exam tabs).
