import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  getStudyCoachMessage,
  type StudyCoachEventType,
  type StudyCoachMode,
} from '../services/study-coach-service';
import type { FocusSession, FocusStopResult } from './chat/useFocusTracking';
import './StudyMode.css';

const FOCUS_SPRINT_MS = 25 * 60 * 1000;
const BREAK_SPRINT_MS = 5 * 60 * 1000;
const NUDGE_MIN_INTERVAL_MS = 45 * 1000;
const MAX_NUDGE_CALLS_PER_SPRINT = 4;
const DISTRACTED_SUSTAINED_MS = 20 * 1000;

type ToastVariant = 'success' | 'warning' | 'info';
type FocusUiState = 'focused' | 'distracted' | null;
type StudyPhase = 'idle' | 'focus_running' | 'focus_complete' | 'break_running' | 'break_complete';

type CoachEntry = {
  id: string;
  mode: StudyCoachMode;
  eventType: StudyCoachEventType;
  text: string;
  createdAtMs: number;
};

type StudyModeProps = {
  userId: string;
  activeFocusSession: FocusSession | null;
  focusBusy: boolean;
  currentFocusState: FocusUiState;
  studyDistractions: number;
  firstDriftOffsetSec: number | null;
  showCalibrationModal: boolean;
  variant?: 'main' | 'sidebar';
  onStartFocus: () => void;
  onStopFocus: () => Promise<FocusStopResult | null>;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function formatClock(ms: number): string {
  const totalSec = Math.max(0, Math.ceil(ms / 1000));
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function formatMmSsFromSec(sec: number | null): string {
  if (sec == null || sec < 0) return '—';
  const minutes = Math.floor(sec / 60);
  const seconds = sec % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function phaseLabel(phase: StudyPhase): string {
  switch (phase) {
    case 'idle':
      return 'Ready to start';
    case 'focus_running':
      return 'Focus sprint in progress';
    case 'focus_complete':
      return 'Focus sprint complete';
    case 'break_running':
      return 'Break in progress';
    case 'break_complete':
      return 'Break complete';
    default:
      return 'Study mode';
  }
}

function buildFallbackCoach(eventType: StudyCoachEventType): string {
  if (eventType === 'distracted_sustained_20s') {
    return 'Pause and reset. Pick one concrete micro-task and return to it now.';
  }
  if (eventType === 'back_in_focus') {
    return 'Nice recovery. Keep momentum with one clear next step.';
  }
  if (eventType === 'last_minute') {
    return 'Final minute. Stay locked in and finish this sprint strong.';
  }
  return 'Stay with the sprint. Small focused steps are enough.';
}

export default function StudyMode({
  userId,
  activeFocusSession,
  focusBusy,
  currentFocusState,
  studyDistractions,
  firstDriftOffsetSec,
  showCalibrationModal,
  variant = 'main',
  onStartFocus,
  onStopFocus,
  showToast,
}: StudyModeProps) {
  const [phase, setPhase] = useState<StudyPhase>('idle');
  const [phaseEndAtMs, setPhaseEndAtMs] = useState<number | null>(null);
  const [phaseRemainingMs, setPhaseRemainingMs] = useState<number>(FOCUS_SPRINT_MS);
  const [sprintIndex, setSprintIndex] = useState<number>(1);
  const [awaitingFocusStart, setAwaitingFocusStart] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [lastStopResult, setLastStopResult] = useState<FocusStopResult | null>(null);
  const [coachFeed, setCoachFeed] = useState<CoachEntry[]>([]);

  const focusSessionIdRef = useRef<string | null>(null);
  const focusStartedAtMsRef = useRef<number | null>(null);
  const phaseRef = useRef<StudyPhase>('idle');
  const focusStateRef = useRef<FocusUiState>(currentFocusState);
  const previousFocusStateRef = useRef<FocusUiState>(null);
  const distractedTimerRef = useRef<number | null>(null);
  const nudgeCallsRef = useRef<number>(0);
  const lastNudgeAtMsRef = useRef<number>(0);
  const lastMinuteNudgeSentRef = useRef<boolean>(false);
  const stopInFlightRef = useRef<boolean>(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    focusStateRef.current = currentFocusState;
  }, [currentFocusState]);

  const appendCoachMessage = useCallback((mode: StudyCoachMode, eventType: StudyCoachEventType, text: string) => {
    const now = Date.now();
    setCoachFeed((prev) => {
      const next: CoachEntry = {
        id: `${now}-${Math.random().toString(16).slice(2)}`,
        mode,
        eventType,
        text,
        createdAtMs: now,
      };
      return [next, ...prev].slice(0, 6);
    });
  }, []);

  const buildCoachMetrics = useCallback((remainingSec: number) => {
    const elapsedSec = Math.max(0, Math.floor((FOCUS_SPRINT_MS - remainingSec * 1000) / 1000));
    return {
      elapsedSec,
      remainingSec,
      distractionCount: studyDistractions,
      firstDriftSec: firstDriftOffsetSec ?? undefined,
    };
  }, [firstDriftOffsetSec, studyDistractions]);

  const requestCoachMessage = useCallback(async (args: {
    mode: StudyCoachMode;
    eventType: StudyCoachEventType;
    elapsedSec: number;
    remainingSec: number;
    focusPercent?: number;
    distractionCount?: number;
    firstDriftSec?: number;
  }) => {
    if (args.mode === 'nudge') {
      const now = Date.now();
      if (nudgeCallsRef.current >= MAX_NUDGE_CALLS_PER_SPRINT) return null;
      if (now - lastNudgeAtMsRef.current < NUDGE_MIN_INTERVAL_MS) return null;
      nudgeCallsRef.current += 1;
      lastNudgeAtMsRef.current = now;
    }

    try {
      const response = await getStudyCoachMessage({
        userId,
        mode: args.mode,
        phase,
        eventType: args.eventType,
        sprintIndex,
        elapsedSec: args.elapsedSec,
        remainingSec: args.remainingSec,
        focusPercent: args.focusPercent,
        distractionCount: args.distractionCount,
        firstDriftSec: args.firstDriftSec,
      });

      const message = response.message?.trim();
      const finalText = message && message.length > 0
        ? message
        : buildFallbackCoach(args.eventType);
      appendCoachMessage(args.mode, args.eventType, finalText);
      return finalText;
    } catch (error) {
      console.error('Study coach request failed:', error);
      const fallback = buildFallbackCoach(args.eventType);
      appendCoachMessage(args.mode, args.eventType, fallback);
      return fallback;
    }
  }, [appendCoachMessage, phase, sprintIndex, userId]);

  const stopFocusSprint = useCallback(async () => {
    if (stopInFlightRef.current || !activeFocusSession) return;
    stopInFlightRef.current = true;
    setIsStopping(true);
    try {
      const stopResult = await onStopFocus();
      if (!stopResult || !stopResult.ok) {
        showToast('Could not finalize sprint. Try stopping focus again.', 'warning');
        return;
      }

      setLastStopResult(stopResult);
      setPhase('focus_complete');
      setPhaseEndAtMs(null);
      setPhaseRemainingMs(0);
    } finally {
      setIsStopping(false);
      stopInFlightRef.current = false;
    }
  }, [activeFocusSession, onStopFocus, showToast]);

  useEffect(() => {
    if (phaseEndAtMs == null) {
      setPhaseRemainingMs(phase === 'break_running' ? BREAK_SPRINT_MS : FOCUS_SPRINT_MS);
      return;
    }

    const tick = () => {
      setPhaseRemainingMs(Math.max(0, phaseEndAtMs - Date.now()));
    };
    tick();
    const timerId = window.setInterval(tick, 250);
    return () => window.clearInterval(timerId);
  }, [phase, phaseEndAtMs]);

  useEffect(() => {
    if (!awaitingFocusStart) return;
    if (showCalibrationModal) return;
    if (focusBusy) return;
    if (!activeFocusSession) {
      setAwaitingFocusStart(false);
    }
  }, [activeFocusSession, awaitingFocusStart, focusBusy, showCalibrationModal]);

  useEffect(() => {
    if (!activeFocusSession) return;
    if (focusSessionIdRef.current === activeFocusSession.id) return;

    focusSessionIdRef.current = activeFocusSession.id;
    focusStartedAtMsRef.current = activeFocusSession.startedAt?.getTime?.() ?? Date.now();
    setAwaitingFocusStart(false);
    setLastStopResult(null);
    setPhase('focus_running');
    setPhaseEndAtMs((focusStartedAtMsRef.current ?? Date.now()) + FOCUS_SPRINT_MS);
    nudgeCallsRef.current = 0;
    lastNudgeAtMsRef.current = 0;
    lastMinuteNudgeSentRef.current = false;
    previousFocusStateRef.current = null;

    const remainingSec = Math.max(
      0,
      Math.floor((((focusStartedAtMsRef.current ?? Date.now()) + FOCUS_SPRINT_MS) - Date.now()) / 1000)
    );
    const timing = buildCoachMetrics(remainingSec);
    void requestCoachMessage({
      mode: 'nudge',
      eventType: 'sprint_start',
      elapsedSec: timing.elapsedSec,
      remainingSec: timing.remainingSec,
      distractionCount: timing.distractionCount,
      firstDriftSec: timing.firstDriftSec,
    });
  }, [activeFocusSession, buildCoachMetrics, requestCoachMessage]);

  useEffect(() => {
    if (activeFocusSession) return;
    focusSessionIdRef.current = null;
    if (phaseRef.current === 'focus_running' && !stopInFlightRef.current) {
      setPhase('focus_complete');
      setPhaseEndAtMs(null);
    }
  }, [activeFocusSession]);

  useEffect(() => {
    if (phase !== 'focus_running') return;
    if (phaseRemainingMs > 0) return;
    if (!activeFocusSession) return;
    void stopFocusSprint();
  }, [activeFocusSession, phase, phaseRemainingMs, stopFocusSprint]);

  useEffect(() => {
    if (phase !== 'focus_running') return;
    if (phaseRemainingMs <= 0 || phaseRemainingMs > 60_000) return;
    if (lastMinuteNudgeSentRef.current) return;
    lastMinuteNudgeSentRef.current = true;

    const remainingSec = Math.max(0, Math.floor(phaseRemainingMs / 1000));
    const timing = buildCoachMetrics(remainingSec);
    void requestCoachMessage({
      mode: 'nudge',
      eventType: 'last_minute',
      elapsedSec: timing.elapsedSec,
      remainingSec: timing.remainingSec,
      distractionCount: timing.distractionCount,
      firstDriftSec: timing.firstDriftSec,
    });
  }, [buildCoachMetrics, phase, phaseRemainingMs, requestCoachMessage]);

  useEffect(() => {
    if (phase !== 'focus_running') {
      if (distractedTimerRef.current != null) {
        window.clearTimeout(distractedTimerRef.current);
        distractedTimerRef.current = null;
      }
      return;
    }

    if (currentFocusState === 'distracted') {
      if (distractedTimerRef.current != null) return;
      distractedTimerRef.current = window.setTimeout(() => {
        distractedTimerRef.current = null;
        if (phaseRef.current !== 'focus_running') return;
        if (focusStateRef.current !== 'distracted') return;
        const remainingSec = Math.max(0, Math.floor(phaseRemainingMs / 1000));
        const timing = buildCoachMetrics(remainingSec);
        void requestCoachMessage({
          mode: 'nudge',
          eventType: 'distracted_sustained_20s',
          elapsedSec: timing.elapsedSec,
          remainingSec: timing.remainingSec,
          distractionCount: timing.distractionCount,
          firstDriftSec: timing.firstDriftSec,
        });
      }, DISTRACTED_SUSTAINED_MS);
      return;
    }

    if (distractedTimerRef.current != null) {
      window.clearTimeout(distractedTimerRef.current);
      distractedTimerRef.current = null;
    }
  }, [buildCoachMetrics, currentFocusState, phase, phaseRemainingMs, requestCoachMessage]);

  useEffect(() => {
    if (phase !== 'focus_running') {
      previousFocusStateRef.current = currentFocusState;
      return;
    }

    if (previousFocusStateRef.current === 'distracted' && currentFocusState === 'focused') {
      const remainingSec = Math.max(0, Math.floor(phaseRemainingMs / 1000));
      const timing = buildCoachMetrics(remainingSec);
      void requestCoachMessage({
        mode: 'nudge',
        eventType: 'back_in_focus',
        elapsedSec: timing.elapsedSec,
        remainingSec: timing.remainingSec,
        distractionCount: timing.distractionCount,
        firstDriftSec: timing.firstDriftSec,
      });
    }
    previousFocusStateRef.current = currentFocusState;
  }, [buildCoachMetrics, currentFocusState, phase, phaseRemainingMs, requestCoachMessage]);

  useEffect(() => {
    if (phase !== 'break_running') return;
    if (phaseRemainingMs > 0) return;
    setPhase('break_complete');
    setPhaseEndAtMs(null);
  }, [phase, phaseRemainingMs]);

  useEffect(() => {
    return () => {
      if (distractedTimerRef.current != null) {
        window.clearTimeout(distractedTimerRef.current);
      }
    };
  }, []);

  const focusStatusText = useMemo(() => {
    if (currentFocusState === 'focused') return 'Focused';
    if (currentFocusState === 'distracted') return 'Distracted';
    return 'Calibrating';
  }, [currentFocusState]);

  const timerTitle = useMemo(() => {
    if (phase === 'break_running') return 'Break Timer';
    return 'Timer';
  }, [phase]);

  const stopDisabled = focusBusy || isStopping || !activeFocusSession;
  const startDisabled = focusBusy || isStopping || awaitingFocusStart || !!activeFocusSession;

  const handleStartFocusSprint = (isNextSprint: boolean) => {
    if (startDisabled) return;
    if (isNextSprint) {
      setSprintIndex((prev) => prev + 1);
    }
    setAwaitingFocusStart(true);
    onStartFocus();
  };

  const handleStartBreak = () => {
    setPhase('break_running');
    setPhaseEndAtMs(Date.now() + BREAK_SPRINT_MS);
  };

  const handleSkipBreak = () => {
    setPhase('break_complete');
    setPhaseEndAtMs(null);
  };

  return (
    <section className={`study-mode ${variant === 'sidebar' ? 'study-mode--sidebar' : ''}`}>
      <div className="study-card study-card-main">
        <div className="study-head">
          <div>
            <h3>Study Mode</h3>
            <p>{phaseLabel(phase)}</p>
          </div>
          <div className="study-phase-chip">Sprint #{sprintIndex}</div>
        </div>

        <div className="study-timer-block">
          <div className="study-timer-label">{timerTitle}</div>
          <div className="study-timer-value">{formatClock(phaseRemainingMs)}</div>
        </div>

        <div className="study-metrics-row">
          <div className={`study-pill ${currentFocusState === 'distracted' ? 'study-pill-warning' : 'study-pill-success'}`}>
            Status: {focusStatusText}
          </div>
          <div className="study-pill">Distractions: {phase === 'focus_complete' ? (lastStopResult?.summary?.distractions ?? lastStopResult?.uiDistractions ?? studyDistractions) : studyDistractions}</div>
        </div>

        <div className="study-actions">
          {phase === 'idle' && (
            <button
              className="study-btn study-btn-primary"
              type="button"
              onClick={() => handleStartFocusSprint(false)}
              disabled={startDisabled}
            >
              {awaitingFocusStart ? 'Waiting for calibration...' : 'Start Focus Sprint'}
            </button>
          )}

          {phase === 'focus_running' && (
            <button
              className="study-btn study-btn-danger"
              type="button"
              onClick={() => void stopFocusSprint()}
              disabled={stopDisabled}
            >
              End Sprint Early
            </button>
          )}

          {phase === 'focus_complete' && (
            <>
              <button className="study-btn" type="button" onClick={handleStartBreak} disabled={focusBusy || isStopping}>
                Start Break
              </button>
              <button
                className="study-btn study-btn-primary"
                type="button"
                onClick={() => handleStartFocusSprint(true)}
                disabled={startDisabled}
              >
                Start Next Sprint
              </button>
            </>
          )}

          {phase === 'break_running' && (
            <button className="study-btn" type="button" onClick={handleSkipBreak}>
              Skip Break
            </button>
          )}

          {phase === 'break_complete' && (
            <button
              className="study-btn study-btn-primary"
              type="button"
              onClick={() => handleStartFocusSprint(true)}
              disabled={startDisabled}
            >
              Start Next Sprint
            </button>
          )}
        </div>
      </div>

      <div className="study-card study-card-coach">
        <h4>Live Coach</h4>
        {coachFeed.length === 0 ? (
          <p className="study-muted">Coach messages will appear during your sprint.</p>
        ) : (
          <div className="study-coach-feed">
            {coachFeed.map((item) => (
              <div key={item.id} className="study-coach-item">
                <div className="study-coach-item-meta">{item.eventType.replace(/_/g, ' ')}</div>
                <div>{item.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
