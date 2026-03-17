# Feature #6 Implementation Tracker

## Feature
**Streaks + lightweight gamification tied to focus minutes**

## Goal
Reward consistent focused study time (not app opens) with:
- daily streaks
- weekly focused-minute goals
- badge unlocks
- simple XP + level progression

## Locked Product Decisions (v1)
- Daily streak threshold: **25 focused minutes/day**
- Weekly goal: **180 focused minutes/week**
- Week boundary: **Monday**
- XP model: **minutes + focus-quality multiplier**
- UI scope: **Dashboard only**
- Rollout: **start fresh** (no historical backfill)

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Definition of Done
- Completed focus sessions can be processed once (idempotent) into gamification progress.
- Streak progression is tied to daily focused-minute totals, not app launches.
- Weekly goal progress increments from focused minutes and marks completion once.
- XP, level, and badge unlocks update from validated session summaries.
- Dashboard shows streaks, weekly goal progress, level/XP, and badge gallery.
- Firestore security and indexes cover new gamification collections.

---

## Step 1: Progression Model + Contracts
- [x] Add server-side constants and helper logic for:
  - [x] day/week keys (timezone-aware day key + Monday week key)
  - [x] XP formula + level formula
  - [x] streak transition rules
  - [x] badge unlock rules
- [x] Define response contract for `POST /gamificationApplyFocusSession`.

## Step 2: Backend Endpoint + Idempotency
- [x] Add `POST /gamificationApplyFocusSession` in `functions/src/index.ts`.
- [x] Validate required body fields: `userId`, `focusSessionId`, `timezone` (normalized).
- [x] Validate focus session ownership and ended status.
- [x] Validate summary ownership and parse focus metrics.
- [x] Clamp focused minutes using session-duration sanity cap.
- [x] Implement transaction-based idempotency with `gamificationSessionAwards/{focusSessionId}`.
- [x] Return `alreadyProcessed` on retries with stable award/profile payload.

## Step 3: Gamification Collections
- [x] `gamificationProfiles/{userId}` (streaks, XP/level, weekly snapshot, badges)
- [x] `gamificationDailyStats/{userId_YYYY-MM-DD}` (daily aggregation + qualification)
- [x] `gamificationWeeklyStats/{userId_YYYY-Www}` (weekly aggregation + completion)
- [x] `gamificationSessionAwards/{focusSessionId}` (immutable processed-session ledger)

## Step 4: Frontend Integration
- [x] Add shared gamification types in `src/types.ts`.
- [x] Add service wrapper `src/services/gamification-service.ts`.
- [x] Call `/gamificationApplyFocusSession` after successful summary persistence in `useFocusTracking`.
- [x] Keep call non-blocking so focus-stop success remains unaffected on gamification failure.

## Step 5: Dashboard UI (v1 scope)
- [x] Subscribe to `gamificationProfiles/{userId}` in `FocusDashboard`.
- [x] Render current streak + longest streak.
- [x] Render level + XP progress bar.
- [x] Render weekly goal progress (`x/180`) with progress bar.
- [x] Render badge gallery (unlocked vs locked).
- [x] Preserve existing focus analytics charts and table behavior.

## Step 6: Security + Indexes
- [x] Update Firestore rules for new gamification collections.
- [x] Enforce owner-read + backend-only writes for profile/daily/weekly/award docs.
- [x] Add indexes for daily and weekly stats query patterns.

## Step 7: Documentation
- [x] Update README endpoint list for `POST /gamificationApplyFocusSession`.
- [x] Document progression formulas, thresholds, and start-fresh rollout policy.

## Step 8: QA Matrix
- [x] Same `focusSessionId` processed twice awards once only.
- [x] Daily totals below 25 do not qualify; crossing 25 in same day qualifies once.
- [x] Consecutive qualified days increase streak; missed day resets next qualified day to 1.
- [x] Multiple sessions in same day aggregate without duplicate streak increments.
- [x] Monday week boundaries roll weekly progress correctly.
- [x] Timezone around local midnight maps to correct day/week key.
- [x] Unauthorized session or summary access returns `403`.
- [ ] Dashboard empty state works before first processed session.
- [x] Emulator regression suite added: `functions/scripts/gamification-emulator-tests.mjs` and runnable via `npm --prefix functions run test:gamification:emulator`.

---

## Active Blockers
- [ ] None currently.

## Change Log
- 2026-03-16: Initial tracker created and implementation completed for backend endpoint, frontend integration, dashboard UI, Firestore rules/indexes, and README updates.
- 2026-03-16: Added and executed emulator QA suite covering idempotency, authorization, daily/weekly progression, timezone boundaries, and no-backfill rollout.
