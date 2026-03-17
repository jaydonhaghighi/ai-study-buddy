import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

const projectId = process.env.GCLOUD_PROJECT || 'demo-ai-study-buddy';
initializeApp({ projectId });
const db = getFirestore();

const functionUrl = `http://127.0.0.1:5001/${projectId}/us-central1/gamificationApplyFocusSession`;

function makeId(prefix) {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, '')}`;
}

function toEpochSec(date) {
  return Math.floor(date.getTime() / 1000);
}

async function clearGamificationDocs(userId) {
  const batches = [
    db.collection('gamificationProfiles').doc(userId),
  ];
  for (const ref of batches) {
    await ref.delete().catch(() => {});
  }

  const groups = ['gamificationDailyStats', 'gamificationWeeklyStats', 'gamificationSessionAwards'];
  for (const group of groups) {
    const snap = await db.collection(group).where('userId', '==', userId).limit(200).get();
    if (snap.empty) continue;
    const batch = db.batch();
    for (const doc of snap.docs) batch.delete(doc.ref);
    await batch.commit();
  }
}

async function seedEndedFocusSession({
  userId,
  focusSessionId,
  startedAt,
  endedAt,
  focusedMs,
  distractedMs = 0,
  focusPercent,
}) {
  await db.collection('focusSessions').doc(focusSessionId).set({
    id: focusSessionId,
    userId,
    status: 'ended',
    source: 'webcam',
    startedAt,
    endedAt,
    createdAt: startedAt,
    updatedAt: endedAt,
  });

  const computedPercent = focusedMs + distractedMs > 0 ? (focusedMs / (focusedMs + distractedMs)) * 100 : 0;
  await db.collection('focusSummaries').doc(focusSessionId).set({
    focusSessionId,
    userId,
    focusedMs,
    distractedMs,
    focusPercent: focusPercent ?? Math.round(computedPercent * 10) / 10,
    endTs: toEpochSec(endedAt),
    createdAt: endedAt,
  });
}

async function callApply({ userId, focusSessionId, timezone }) {
  const res = await fetch(functionUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, focusSessionId, timezone }),
  });

  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { parseError: text };
  }

  return { status: res.status, body };
}

async function runScenario(name, fn) {
  try {
    await fn();
    console.log(`PASS: ${name}`);
  } catch (error) {
    console.error(`FAIL: ${name}`);
    console.error(error instanceof Error ? error.stack || error.message : error);
    throw error;
  }
}

async function testIdempotencyAndAuthorization() {
  const userId = makeId('user_idem');
  const intruderId = makeId('user_intruder');
  await clearGamificationDocs(userId);

  const focusSessionId = makeId('fs_idem');
  const startedAt = new Date('2026-03-16T13:00:00.000Z');
  const endedAt = new Date('2026-03-16T13:30:00.000Z');
  await seedEndedFocusSession({
    userId,
    focusSessionId,
    startedAt,
    endedAt,
    focusedMs: 25 * 60 * 1000,
    distractedMs: 5 * 60 * 1000,
    focusPercent: 83.3,
  });

  const first = await callApply({
    userId,
    focusSessionId,
    timezone: 'America/New_York',
  });
  assert.equal(first.status, 200, 'first call should succeed');
  assert.equal(first.body?.alreadyProcessed, false, 'first call should process session');
  assert.equal(first.body?.award?.focusedMinutes, 25, 'first award focused minutes mismatch');
  assert.equal(first.body?.award?.xpGain, 28, 'first award XP mismatch (25 * 1.10 rounded)');
  assert.equal(first.body?.profile?.currentStreakDays, 1, 'streak should start at 1 once day qualifies');

  const second = await callApply({
    userId,
    focusSessionId,
    timezone: 'America/New_York',
  });
  assert.equal(second.status, 200, 'second call should succeed');
  assert.equal(second.body?.alreadyProcessed, true, 'second call should be idempotent');
  assert.equal(second.body?.profile?.totalXp, first.body?.profile?.totalXp, 'idempotent replay must not re-award XP');

  const unauthorized = await callApply({
    userId: intruderId,
    focusSessionId,
    timezone: 'America/New_York',
  });
  assert.equal(unauthorized.status, 403, 'unauthorized user should receive 403');
}

async function testDailyThresholdAndSameDayAggregation() {
  const userId = makeId('user_day');
  await clearGamificationDocs(userId);

  const tz = 'UTC';
  const day = '2026-03-17';

  const sessionA = makeId('fs_day_a');
  await seedEndedFocusSession({
    userId,
    focusSessionId: sessionA,
    startedAt: new Date(`${day}T08:00:00.000Z`),
    endedAt: new Date(`${day}T08:20:00.000Z`),
    focusedMs: 10 * 60 * 1000,
    distractedMs: 10 * 60 * 1000,
  });

  const sessionB = makeId('fs_day_b');
  await seedEndedFocusSession({
    userId,
    focusSessionId: sessionB,
    startedAt: new Date(`${day}T10:00:00.000Z`),
    endedAt: new Date(`${day}T10:20:00.000Z`),
    focusedMs: 15 * 60 * 1000,
    distractedMs: 5 * 60 * 1000,
  });

  const sessionC = makeId('fs_day_c');
  await seedEndedFocusSession({
    userId,
    focusSessionId: sessionC,
    startedAt: new Date(`${day}T12:00:00.000Z`),
    endedAt: new Date(`${day}T12:15:00.000Z`),
    focusedMs: 10 * 60 * 1000,
    distractedMs: 5 * 60 * 1000,
  });

  const first = await callApply({ userId, focusSessionId: sessionA, timezone: tz });
  assert.equal(first.status, 200);
  assert.equal(first.body?.profile?.currentStreakDays, 0, 'day should not qualify at 10 minutes');

  const second = await callApply({ userId, focusSessionId: sessionB, timezone: tz });
  assert.equal(second.status, 200);
  assert.equal(second.body?.profile?.currentStreakDays, 1, 'day should qualify when crossing 25 minutes');

  const third = await callApply({ userId, focusSessionId: sessionC, timezone: tz });
  assert.equal(third.status, 200);
  assert.equal(third.body?.profile?.currentStreakDays, 1, 'same day extra sessions should not increment streak again');
}

async function testConsecutiveAndResetStreak() {
  const userId = makeId('user_streak');
  await clearGamificationDocs(userId);

  const sessions = [
    { sid: makeId('fs_streak_1'), start: '2026-03-10T08:00:00.000Z', end: '2026-03-10T08:30:00.000Z' },
    { sid: makeId('fs_streak_2'), start: '2026-03-11T08:00:00.000Z', end: '2026-03-11T08:30:00.000Z' },
    { sid: makeId('fs_streak_4'), start: '2026-03-13T08:00:00.000Z', end: '2026-03-13T08:30:00.000Z' },
  ];

  for (const session of sessions) {
    await seedEndedFocusSession({
      userId,
      focusSessionId: session.sid,
      startedAt: new Date(session.start),
      endedAt: new Date(session.end),
      focusedMs: 25 * 60 * 1000,
      distractedMs: 5 * 60 * 1000,
    });
  }

  const r1 = await callApply({ userId, focusSessionId: sessions[0].sid, timezone: 'UTC' });
  const r2 = await callApply({ userId, focusSessionId: sessions[1].sid, timezone: 'UTC' });
  const r3 = await callApply({ userId, focusSessionId: sessions[2].sid, timezone: 'UTC' });

  assert.equal(r1.status, 200);
  assert.equal(r2.status, 200);
  assert.equal(r3.status, 200);
  assert.equal(r2.body?.profile?.currentStreakDays, 2, 'second consecutive day should increment streak to 2');
  assert.equal(r3.body?.profile?.currentStreakDays, 1, 'streak should reset to 1 after a missed day');
  assert.equal(r3.body?.profile?.longestStreakDays, 2, 'longest streak should be retained');
}

async function testMondayWeekBoundary() {
  const userId = makeId('user_week');
  await clearGamificationDocs(userId);

  const sundaySession = makeId('fs_week_sun');
  const mondaySession = makeId('fs_week_mon');

  await seedEndedFocusSession({
    userId,
    focusSessionId: sundaySession,
    startedAt: new Date('2026-03-15T18:00:00.000Z'),
    endedAt: new Date('2026-03-15T20:00:00.000Z'),
    focusedMs: 100 * 60 * 1000,
  });

  await seedEndedFocusSession({
    userId,
    focusSessionId: mondaySession,
    startedAt: new Date('2026-03-16T18:00:00.000Z'),
    endedAt: new Date('2026-03-16T20:00:00.000Z'),
    focusedMs: 100 * 60 * 1000,
  });

  const sunday = await callApply({ userId, focusSessionId: sundaySession, timezone: 'UTC' });
  const monday = await callApply({ userId, focusSessionId: mondaySession, timezone: 'UTC' });

  assert.equal(sunday.status, 200);
  assert.equal(monday.status, 200);
  assert.notEqual(sunday.body?.award?.weekKey, monday.body?.award?.weekKey, 'Sunday and Monday should map to different week keys');
  assert.equal(monday.body?.profile?.weeklyGoal?.focusedMinutes, 100, 'new Monday week should start with that week minutes only');
}

async function testTimezoneMidnightBoundary() {
  const userId = makeId('user_tz');
  await clearGamificationDocs(userId);

  const beforeMidnight = makeId('fs_tz_before');
  const afterMidnight = makeId('fs_tz_after');

  await seedEndedFocusSession({
    userId,
    focusSessionId: beforeMidnight,
    startedAt: new Date('2026-03-16T03:00:00.000Z'),
    endedAt: new Date('2026-03-16T03:30:00.000Z'),
    focusedMs: 25 * 60 * 1000,
  });

  await seedEndedFocusSession({
    userId,
    focusSessionId: afterMidnight,
    startedAt: new Date('2026-03-16T04:00:00.000Z'),
    endedAt: new Date('2026-03-16T04:30:00.000Z'),
    focusedMs: 25 * 60 * 1000,
  });

  const first = await callApply({ userId, focusSessionId: beforeMidnight, timezone: 'America/New_York' });
  const second = await callApply({ userId, focusSessionId: afterMidnight, timezone: 'America/New_York' });

  assert.equal(first.status, 200);
  assert.equal(second.status, 200);
  assert.equal(first.body?.award?.dayKey, '2026-03-15', '03:30Z should map to previous local day in New York');
  assert.equal(second.body?.award?.dayKey, '2026-03-16', '04:30Z should map to next local day in New York');
  assert.equal(second.body?.profile?.currentStreakDays, 2, 'back-to-back local days should form a 2-day streak');
}

async function testStartFreshNoBackfill() {
  const userId = makeId('user_fresh');
  await clearGamificationDocs(userId);

  const historicalSession = makeId('fs_hist');
  const newSession = makeId('fs_new');

  await seedEndedFocusSession({
    userId,
    focusSessionId: historicalSession,
    startedAt: new Date('2026-02-10T08:00:00.000Z'),
    endedAt: new Date('2026-02-10T09:00:00.000Z'),
    focusedMs: 60 * 60 * 1000,
  });

  await seedEndedFocusSession({
    userId,
    focusSessionId: newSession,
    startedAt: new Date('2026-03-18T08:00:00.000Z'),
    endedAt: new Date('2026-03-18T08:30:00.000Z'),
    focusedMs: 30 * 60 * 1000,
  });

  const result = await callApply({ userId, focusSessionId: newSession, timezone: 'UTC' });
  assert.equal(result.status, 200);
  assert.equal(result.body?.profile?.totalFocusedMinutes, 30, 'unprocessed historical sessions should not auto-backfill progress');
}

async function main() {
  console.log('Running gamification emulator tests against:', functionUrl);

  await runScenario('Idempotency + authorization', testIdempotencyAndAuthorization);
  await runScenario('Daily threshold + same-day aggregation', testDailyThresholdAndSameDayAggregation);
  await runScenario('Consecutive streak + reset behavior', testConsecutiveAndResetStreak);
  await runScenario('Monday week boundary handling', testMondayWeekBoundary);
  await runScenario('Timezone local-midnight handling', testTimezoneMidnightBoundary);
  await runScenario('Start-fresh rollout (no backfill)', testStartFreshNoBackfill);

  console.log('All gamification emulator scenarios passed.');
}

main().catch((error) => {
  console.error('Gamification emulator tests failed.');
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exit(1);
});
