# Feature #2 Implementation Tracker

## Feature
**Configurable unfocused nudge timing (sound alert)**

## Goal
Let each student choose how long they can remain distracted during an active focus session before the app plays a sound nudge.

## Current Implementation Snapshot
- The settings menu now includes a `Focus alerts` action that opens a dedicated modal.
- Preferences persist in Firestore under `userSettings/{userId}` with a v1 default of `1 minute`, `sound on`, and `60%` volume.
- `src/components/chat/useFocusTracking.ts` now plays one delayed nudge per continuous distraction streak and resets that timer on refocus, session stop, sign-out, or settings changes.
- `src/components/StudyMode.tsx` still keeps its separate 20-second coach-text timer; audio timing is intentionally independent.

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Definition of Done
- A signed-in user can configure an unfocused sound-nudge delay from `1-15` minutes.
- The selected value persists after refresh and sign-in on another device.
- During an active focus session, the app plays an audio nudge only after the user has remained distracted for the configured duration.
- The nudge timer resets correctly when the user refocuses, stops the session, signs out, or starts a new session.
- The existing immediate transition beep path is either removed for `distracted` or clearly separated so users do not hear duplicate alerts.
- Firestore rules allow owner-scoped reads/writes for the new settings doc and reject cross-user access.

---

## Step 1: Product Contract + Baseline Reconciliation
- [x] Trigger only while a focus session is active.
- [x] Measure continuous distracted time, not cumulative distracted time across a sprint.
- [x] Play one sound per distraction streak; require a refocus before another timed nudge can fire.
- [x] Ship `sound on/off` and `volume` in the same pass.
- [x] Resolve the roadmap/default mismatch by standardizing v1 on a `1 minute` default.

## Step 2: Data Model + Shared Types
- [x] Add shared frontend type in `src/types.ts` for `nudgeDelayMinutes`, `soundEnabled`, `volume`, and `updatedAt`.
- [x] Create Firestore doc shape under `userSettings/{userId}` with owner metadata.
- [ ] Add a lightweight `localStorage` cache for faster first paint if needed later.

## Step 3: Firestore Rules + Index Review
- [x] Update `firestore.rules` for `userSettings/{userId}` with owner-scoped read/create/update/delete rules.
- [x] Confirm no new composite index is needed for single-doc reads.

## Step 4: Frontend Settings Service + Hook
- [x] Add `src/services/focus-alert-settings-service.ts` for loading/parsing/saving focus-alert preferences.
- [x] Add `src/components/chat/useFocusAlertSettings.ts` for subscription, defaulting, and save state.
- [x] Keep the contract narrow so it can be reused by both chat and study-mode surfaces.

## Step 5: Settings UI
- [x] Extend the current settings menu in `Chat.tsx` and `ChatSidebar.tsx` with a `Focus alerts` action.
- [x] Build a compact modal instead of fitting sliders into the existing popover menu.
- [x] Add the main `1-15` minute delay control and helper copy.
- [x] Add `Sound alerts` and `Volume` controls with preview.
- [x] Show save feedback via existing toast messaging.

## Step 6: Audio/Nudge Engine Refactor
- [x] Extract the hardcoded beep logic from `Chat.tsx` into `src/components/chat/useFocusAlertAudio.ts`.
- [x] Keep the refocus success sound and remove the immediate distracted transition beep.
- [x] Parameterize volume and keep graceful browser-audio failure handling.

## Step 7: Focus Timer Integration
- [x] Implement the sustained-distraction timer in `src/components/chat/useFocusTracking.ts`.
- [x] Add refs for distraction start time, active timeout id, and per-streak fire tracking.
- [x] Arm on distraction, fire after `nudgeDelayMinutes * 60_000`, cancel on cleanup/refocus, and recompute when settings change mid-streak.

## Step 8: Study Mode Coordination
- [x] Keep `StudyMode.tsx` coach timing independent from the audio nudge delay.
- [ ] Rename or document `DISTRACTED_SUSTAINED_MS` as coach-specific in a follow-up cleanup.
- [ ] Surface a small in-study visual event when the timed sound nudge fires if that proves useful.

## Step 9: QA Matrix
- [x] Automated verification: `npm run build`
- [x] Automated verification: `npm run lint`
- [ ] Manual browser QA for delay variants, per-streak reset behavior, and mid-session settings changes.
- [ ] Emulator/security-rule tests for cross-user access on `userSettings`.

## Step 10: Rollout
- [x] Update roadmap/docs so the default behavior described in docs matches the real app.
- [ ] Monitor real-user feedback on whether the `1 minute` default feels too aggressive or too lenient.

---

## Suggested Build Order
1. Step 1 (lock product contract and reconcile the default mismatch)
2. Step 2 + Step 3 (settings data model and security)
3. Step 4 + Step 5 (settings hook and UI)
4. Step 6 + Step 7 (audio refactor and timed nudge integration)
5. Step 8 + Step 9 (Study Mode alignment and QA)
6. Step 10 (rollout/docs)

## Active Blockers
- [ ] None currently.

## Change Log
- 2026-03-27: Initial Feature #2 execution plan created.
- 2026-03-27: Implemented Firestore-backed focus-alert settings, modal UI, configurable sound/volume controls, and delayed unfocused nudge timing in `useFocusTracking`.
