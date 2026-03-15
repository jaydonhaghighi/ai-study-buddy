# Feature #5 Implementation Tracker

## Feature
**Upload/import course material, then chat with citations**

## Goal
Let users upload course materials (`pdf`, `docx`, Excel, slides, `txt`, images), ask questions grounded in those materials, and receive answers with citations.

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Definition of Done
- Users can upload and manage course materials for a selected study context.
- Supported file types: `pdf`, `docx`, `xlsx/xls`, `pptx/ppt`, `txt`, and common images (`png`, `jpg`, `jpeg`, `webp`).
- Uploaded material is parsed, chunked, and indexed.
- Chat answers are grounded in retrieved context.
- AI answers include citations mapped to source locations (`page`, `sheet`, `slide`, `line`, or image region).
- Authorization and ownership are enforced for files, metadata, retrieval, and chat.

---

## Phase 0: Scope + Design
- [x] Confirm supported types for initial release: `pdf`, `docx`, `xlsx/xls`, `pptx/ppt`, `txt`, `png/jpg/jpeg/webp`.
- [x] Confirm retrieval scope: current user + selected course/session/chat context only.
- [x] Confirm citation format in UI (inline chips, expandable source details, or footer list).
- [x] Define fallback behavior when no relevant docs are found.
- [x] Define parser strategy per type (native text extraction vs OCR).
- [ ] Write architecture note for ingestion + retrieval + citation flow.

## Phase 1: Data Model + Security
- [x] Create Firestore collection schema for `courseMaterials`:
  - [x] `id`, `userId`, `courseId`, `sessionId`, `chatId?`
  - [x] `filename`, `mimeType`, `extension`, `sizeBytes`, `storagePath`
  - [x] `fileType` (`pdf|docx|spreadsheet|slides|txt|image`)
  - [x] `status` (`uploaded|processing|indexed|failed`)
  - [x] `chunkCount`, `errorMessage?`, `processingMs?`, timestamps
- [x] Extend message schema for AI citation metadata:
  - [x] `citations[]` with `materialId`, `chunkId`, `locationType`, `locationLabel`, `snippet`, `score?`
- [ ] Add/adjust Firestore rules for `courseMaterials` and citation-safe message reads.
- [ ] Add Firebase Storage rules for per-user isolation under `users/{uid}/courseMaterials/...`.

## Phase 2: Frontend Material Management UI
- [x] Build upload panel inside right sidebar (`ChatPreviewSidebar`) using existing file-zone styles.
- [x] Support drag-and-drop + file picker for all supported file types.
- [x] Add client-side validation for MIME/extension and file-size limits.
- [x] Show upload states (uploading, processing, indexed, failed).
- [x] Show material list with file name, size, and status.
- [x] Add delete action (soft delete or hard delete based on chosen policy).

## Phase 3: Frontend Services + State Wiring
- [x] Add material service functions (upload/list/delete/status updates).
- [x] Wire material state into chat screen context (`Chat.tsx` + chat hooks).
- [x] Persist selected/active materials per chat/session.
- [x] Add optimistic UI updates and rollback on failures.
- [x] Add toast/error messaging for upload and indexing failures.

## Phase 4: Ingestion Flow (Backend)
- [x] Add backend endpoint/function for indexing uploaded material.
- [x] Parse file content by type:
  - [x] PDF via `pdf-parse`
  - [x] DOCX via document text extractor
  - [x] Excel via workbook parser (sheet-wise text flattening)
  - [x] Slides via presentation parser (slide-wise extraction)
  - [x] TXT via plain UTF-8 read
  - [x] Images via OCR service/library
- [x] Normalize extracted content into common `Document` chunks with source location metadata.
- [x] Chunk text (custom chunker) with tuned chunk size + overlap.
- [x] Attach chunk metadata (`materialId`, `fileName`, `locationType`, `locationLabel`, `chunkIndex`, `course/session/chat refs`).
- [x] Index chunks with retriever-compatible structure.
- [x] Save indexing result + status back to `courseMaterials`.

## Phase 5: Retrieval + Grounded Chat
- [x] Update `/chat` flow to retrieve relevant chunks before generation.
- [x] Restrict retrieval to user-owned, context-appropriate materials.
- [x] Ground prompt with strict instruction: use provided context only; do not hallucinate.
- [x] Handle no-context and low-relevance cases cleanly.
- [x] Keep existing persistent session behavior intact.

## Phase 6: Citation Generation + SSE Contract
- [x] Define backend response shape for final SSE event including citations.
- [x] Map retrieved chunks to citation IDs and source details.
- [x] Ensure AI output references citation IDs consistently.
- [x] Return `fullText + citations + model + sessionId` in final stream event.
- [x] Keep backward compatibility for older clients (no citations expected).

## Phase 7: Frontend Citation Rendering
- [x] Extend client stream parser (`genkit-service`) to read final citations payload.
- [x] Extend `Message` type + Firestore write path to store `citations` on AI messages.
- [x] Render citations in `ChatMessageList` per AI message.
- [x] Add citation UX details (source label with `page/sheet/slide/line/image`, snippet preview, external link if available).
- [x] Ensure markdown rendering and citation UI coexist cleanly.

## Phase 8: QA, Hardening, and Rollout
- [ ] Test matrix:
  - [ ] No uploaded files
  - [ ] Single PDF
  - [ ] Single DOCX
  - [ ] Single Excel workbook (multi-sheet)
  - [ ] Single slide deck
  - [ ] Single TXT
  - [ ] Single image (typed text)
  - [ ] Single image (scanned/noisy text)
  - [ ] Multiple PDFs in same session
  - [ ] Mixed file types in same session
  - [ ] Large PDF
  - [ ] Unsupported file type
  - [ ] Unauthorized access attempt
  - [ ] Deleted file referenced by old citation
- [ ] Validate security rules with emulator tests.
- [ ] Validate retrieval quality and citation accuracy with sample course docs.
- [ ] Add logging/observability for indexing and retrieval failures.
- [ ] Ship behind a feature flag if needed.

---

## Stretch Goals (Post-MVP)
- [ ] Add reranking / two-stage retrieval.
- [ ] Add per-material toggle (include/exclude in retrieval).
- [ ] Add citation confidence and “open source excerpt” panel.
- [ ] Add ingestion queue/background processing for large files.
- [ ] Add advanced OCR for handwriting and diagrams.
- [ ] Add table-aware retrieval for spreadsheet-heavy coursework.

## Active Blockers
- [ ] None currently.

## Change Log
- 2026-03-15: Initial tracker created.
- 2026-03-15: Scope expanded to support DOCX, Excel, slides, TXT, and image uploads with OCR + location-aware citations.
- 2026-03-15: Implemented end-to-end MVP (upload panel, indexing endpoints, retrieval in `/chat`, SSE citations, citation UI rendering).
- 2026-03-15: Hardened Firestore rules with owner-scoped access and backend-only protections for `material_chunks` and `genkit_sessions`.
