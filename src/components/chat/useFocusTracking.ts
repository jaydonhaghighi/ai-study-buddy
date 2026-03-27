import { useEffect, useRef, useState } from 'react';
import { User } from 'firebase/auth';
import {
  collection,
  doc,
  onSnapshot,
  query,
  serverTimestamp,
  setDoc,
  Timestamp,
  where,
} from 'firebase/firestore';
import { db } from '../../firebase-config';
import { startFocusSession, stopFocusSession } from '../../services/focus-service';
import { applyGamificationForFocusSession } from '../../services/gamification-service';
import { LaptopFocusTracker } from '../../services/laptop-focus-tracker';
import {
  InferenceFocusTracker,
  type InferencePredictionPayload,
} from '../../services/inference-focus-tracker';
import type { FocusAlertSettings } from '../../types';
import { acquireWebcamStream } from '../../services/webcam-manager';

type ToastVariant = 'success' | 'warning' | 'info';
type TrackerSource = 'ml_inference_api' | 'laptop_webcam';

export type FocusTrackerSummary = {
  startTs: number;
  endTs: number;
  focusedMs: number;
  distractedMs: number;
  distractions: number;
  focusPercent: number;
  attentionLabelCounts: Record<string, number>;
  [key: string]: unknown;
};

export type FocusStopResult = {
  ok: boolean;
  focusSessionId: string;
  source: TrackerSource | null;
  summary: FocusTrackerSummary | null;
  uiDistractions: number;
  firstDriftOffsetSec: number | null;
};

type FocusTrackerRuntime = {
  start(): Promise<void>;
  stop(): Promise<FocusTrackerSummary>;
};

export type FocusSession = {
  id: string;
  userId: string;
  status: string;
  courseId?: string | null;
  sessionId?: string | null;
  mode?: 'study_mode' | 'exam_simulation' | null;
  chatId?: string | null;
  examSimulationId?: string | null;
  startedAt?: Date | null;
};

export type FocusStartOptions = {
  courseId?: string;
  sessionId?: string;
  mode?: 'study_mode' | 'exam_simulation';
  chatId?: string;
  examSimulationId?: string;
  onStarted?: (focusSessionId: string) => Promise<void> | void;
};

const FAST_UI_CONFIDENCE_THRESHOLD = 0.45;
const FAST_UI_DISTRACT_HOLD_MS = 450;
const FAST_UI_REFOCUS_HOLD_MS = 250;

type UseFocusTrackingParams = {
  user: User | null;
  expandedCourseId: string | null;
  expandedSessionId: string | null;
  focusAlertSettings: FocusAlertSettings;
  showToast: (message: string, variant?: ToastVariant) => void;
  playFocusRecoverySound: () => void;
  playDistractedNudgeSound: () => void;
};

export function useFocusTracking({
  user,
  expandedCourseId,
  expandedSessionId,
  focusAlertSettings,
  showToast,
  playFocusRecoverySound,
  playDistractedNudgeSound,
}: UseFocusTrackingParams) {
  const [activeFocusSession, setActiveFocusSession] = useState<FocusSession | null>(null);
  const [focusBusy, setFocusBusy] = useState(false);
  const [isLocalTrackerRunning, setIsLocalTrackerRunning] = useState(false);
  const [activeTrackerLabel, setActiveTrackerLabel] = useState<string | null>(null);
  const [lastPose, setLastPose] = useState<InferencePredictionPayload | null>(null);
  const [focusElapsedMs, setFocusElapsedMs] = useState<number>(0);
  const [currentFocusState, setCurrentFocusState] = useState<'focused' | 'distracted' | null>(null);
  const [studyDistractions, setStudyDistractions] = useState<number>(0);
  const [firstDriftOffsetSec, setFirstDriftOffsetSec] = useState<number | null>(null);
  const [showCalibrationModal, setShowCalibrationModal] = useState(false);
  const [pendingFocusStart, setPendingFocusStart] = useState<{
    courseId?: string;
    sessionId?: string;
    mode?: 'study_mode' | 'exam_simulation';
    chatId?: string;
    examSimulationId?: string;
    onStarted?: (focusSessionId: string) => Promise<void> | void;
  } | null>(null);

  const localFocusTrackerRef = useRef<FocusTrackerRuntime | null>(null);
  const trackerSourceRef = useRef<TrackerSource | null>(null);
  const trackerReleaseRef = useRef<null | (() => void)>(null);
  const focusStartLocalMsRef = useRef<number | null>(null);
  const activeFocusSessionRef = useRef<FocusSession | null>(null);

  const announcedFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateSinceMsRef = useRef<number | null>(null);
  const lastAnnounceMsRef = useRef<number>(0);
  const uiFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const uiDistractionsRef = useRef<number>(0);
  const firstDriftOffsetSecRef = useRef<number | null>(null);
  const distractedSinceMsRef = useRef<number | null>(null);
  const distractedNudgeTimerRef = useRef<number | null>(null);
  const distractedNudgePlayedRef = useRef<boolean>(false);

  const clearPendingDistractedNudge = () => {
    if (distractedNudgeTimerRef.current != null) {
      window.clearTimeout(distractedNudgeTimerRef.current);
      distractedNudgeTimerRef.current = null;
    }
  };

  const announceFocusState = (nextState: 'focused' | 'distracted') => {
    if (announcedFocusStateRef.current === nextState) return;
    const now = Date.now();
    if (now - lastAnnounceMsRef.current < 800) return;

    announcedFocusStateRef.current = nextState;
    lastAnnounceMsRef.current = now;

    if (nextState === 'distracted') {
      showToast('You are distracted', 'warning');
    } else {
      showToast("You're back in focus", 'success');
      playFocusRecoverySound();
    }
  };

  const resetUiDistractionCounter = () => {
    clearPendingDistractedNudge();
    uiFocusStateRef.current = null;
    uiDistractionsRef.current = 0;
    firstDriftOffsetSecRef.current = null;
    distractedSinceMsRef.current = null;
    distractedNudgePlayedRef.current = false;
    setCurrentFocusState(null);
    setStudyDistractions(0);
    setFirstDriftOffsetSec(null);
    candidateFocusStateRef.current = null;
    candidateSinceMsRef.current = null;
    announcedFocusStateRef.current = null;
    lastAnnounceMsRef.current = 0;
  };

  const applyUiFocusState = (nextState: 'focused' | 'distracted') => {
    const prev = uiFocusStateRef.current;
    if (prev === 'focused' && nextState === 'distracted') {
      distractedSinceMsRef.current = Date.now();
      distractedNudgePlayedRef.current = false;
      uiDistractionsRef.current += 1;
      setStudyDistractions(uiDistractionsRef.current);
      if (firstDriftOffsetSecRef.current == null) {
        const startedAtMs = activeFocusSession?.startedAt?.getTime?.() ?? focusStartLocalMsRef.current ?? Date.now();
        const offsetSec = Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000));
        firstDriftOffsetSecRef.current = offsetSec;
        setFirstDriftOffsetSec(offsetSec);
      }
    }
    if (nextState === 'focused') {
      clearPendingDistractedNudge();
      distractedSinceMsRef.current = null;
      distractedNudgePlayedRef.current = false;
    }
    uiFocusStateRef.current = nextState;
    setCurrentFocusState(nextState);
  };

  useEffect(() => {
    activeFocusSessionRef.current = activeFocusSession;
  }, [activeFocusSession]);

  useEffect(() => {
    if (!user) {
      setActiveFocusSession(null);
      return;
    }

    const q = query(
      collection(db, 'focusSessions'),
      where('userId', '==', user.uid),
      where('status', '==', 'active')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      if (snapshot.empty) {
        setActiveFocusSession(null);
        return;
      }
      const doc0 = snapshot.docs[0];
      const data = doc0.data() as any;
      setActiveFocusSession({
        id: doc0.id,
        ...data,
        mode: data.mode === 'study_mode' || data.mode === 'exam_simulation' ? data.mode : null,
        chatId: typeof data.chatId === 'string' ? data.chatId : null,
        examSimulationId: typeof data.examSimulationId === 'string' ? data.examSimulationId : null,
        startedAt: data.startedAt instanceof Timestamp ? data.startedAt.toDate() : null,
      } as FocusSession);
    });

    return () => unsubscribe();
  }, [user]);

  useEffect(() => {
    return () => {
      clearPendingDistractedNudge();
      const tracker = localFocusTrackerRef.current;
      localFocusTrackerRef.current = null;
      if (trackerReleaseRef.current) {
        trackerReleaseRef.current();
        trackerReleaseRef.current = null;
      }
      if (tracker) {
        void tracker.stop().catch((error) => {
          console.error('Error stopping local focus tracker during cleanup:', error);
        });
      }
    };
  }, []);

  useEffect(() => {
    if (activeFocusSession) return;
    trackerSourceRef.current = null;
    setActiveTrackerLabel(null);
    setLastPose(null);
    setFocusElapsedMs(0);
    uiFocusStateRef.current = null;
    uiDistractionsRef.current = 0;
    firstDriftOffsetSecRef.current = null;
    setCurrentFocusState(null);
    setStudyDistractions(0);
    setFirstDriftOffsetSec(null);
    candidateFocusStateRef.current = null;
    candidateSinceMsRef.current = null;
    announcedFocusStateRef.current = null;
    lastAnnounceMsRef.current = 0;
    focusStartLocalMsRef.current = null;
    const tracker = localFocusTrackerRef.current;
    if (!tracker) return;
    localFocusTrackerRef.current = null;
    setIsLocalTrackerRunning(false);
    void tracker.stop().catch((error) => {
      console.error('Error stopping local focus tracker after session ended:', error);
    });
  }, [activeFocusSession]);

  useEffect(() => {
    if (!activeFocusSession) return;
    const startMs =
      activeFocusSession.startedAt?.getTime?.() ??
      focusStartLocalMsRef.current ??
      Date.now();

    const tick = () => {
      setFocusElapsedMs(Date.now() - startMs);
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [activeFocusSession]);

  useEffect(() => {
    if (!activeFocusSession || currentFocusState !== 'distracted') {
      clearPendingDistractedNudge();
      return;
    }

    if (distractedSinceMsRef.current == null) {
      distractedSinceMsRef.current = Date.now();
    }

    if (!focusAlertSettings.soundEnabled || distractedNudgePlayedRef.current) {
      clearPendingDistractedNudge();
      return;
    }

    const delayMs = Math.max(1, focusAlertSettings.nudgeDelayMinutes) * 60 * 1000;
    const elapsedMs = Date.now() - distractedSinceMsRef.current;
    const remainingMs = Math.max(0, delayMs - elapsedMs);

    clearPendingDistractedNudge();
    distractedNudgeTimerRef.current = window.setTimeout(() => {
      distractedNudgeTimerRef.current = null;
      if (!activeFocusSessionRef.current) return;
      if (uiFocusStateRef.current !== 'distracted') return;
      if (distractedNudgePlayedRef.current) return;
      distractedNudgePlayedRef.current = true;
      playDistractedNudgeSound();
    }, remainingMs);

    return () => {
      clearPendingDistractedNudge();
    };
  }, [
    activeFocusSession,
    currentFocusState,
    focusAlertSettings.nudgeDelayMinutes,
    focusAlertSettings.soundEnabled,
    playDistractedNudgeSound,
  ]);

  const handleStartFocus = (options?: FocusStartOptions) => {
    if (!user) return;
    const courseId = (options?.courseId ?? expandedCourseId) || undefined;
    const sessionId = (options?.sessionId ?? expandedSessionId) || undefined;
    setPendingFocusStart({
      courseId,
      sessionId,
      mode: options?.mode,
      chatId: options?.chatId,
      examSimulationId: options?.examSimulationId,
      onStarted: options?.onStarted,
    });
    setShowCalibrationModal(true);
  };

  const hideCalibrationModal = () => {
    setShowCalibrationModal(false);
  };

  const cancelCalibration = () => {
    setShowCalibrationModal(false);
    setPendingFocusStart(null);
  };

  const startFocusAfterCalibration = async () => {
    if (!user) return;
    if (!pendingFocusStart) return;

    setFocusBusy(true);
    try {
      resetUiDistractionCounter();
      uiFocusStateRef.current = 'focused';
      setCurrentFocusState('focused');
      const res = await startFocusSession({
        userId: user.uid,
        courseId: pendingFocusStart.courseId,
        sessionId: pendingFocusStart.sessionId,
        mode: pendingFocusStart.mode,
        chatId: pendingFocusStart.chatId,
        examSimulationId: pendingFocusStart.examSimulationId,
      });
      focusStartLocalMsRef.current = Date.now();
      if (pendingFocusStart.onStarted) {
        try {
          await pendingFocusStart.onStarted(res.focusSessionId);
        } catch (callbackError) {
          console.warn('Linked focus-start callback failed:', callbackError);
          showToast('Focus tracking started, but the linked exam flow could not continue.', 'warning');
        }
      }

      try {
        let tracker: FocusTrackerRuntime | null = null;
        let trackerSource: TrackerSource = 'ml_inference_api';
        let label = 'Server ML inference';

        try {
          const acquired = await acquireWebcamStream();
          trackerReleaseRef.current = acquired.release;
          const remoteTracker = new InferenceFocusTracker({
            stream: acquired.stream,
            onPrediction: (p) => {
              setLastPose((prev) => {
                if (!prev) return p;
                if (
                  prev.smoothed_label !== p.smoothed_label ||
                  prev.smoothed_confidence !== p.smoothed_confidence ||
                  prev.state !== p.state
                ) {
                  return p;
                }
                return prev;
              });

              if (p.transitioned) {
                const transitionedState: 'focused' | 'distracted' | null =
                  p.state === 'distracted'
                    ? 'distracted'
                    : p.state === 'focused'
                      ? 'focused'
                      : null;
                if (transitionedState) {
                  applyUiFocusState(transitionedState);
                  announceFocusState(transitionedState);
                  candidateFocusStateRef.current = null;
                  candidateSinceMsRef.current = null;
                }
              }

              const smoothedConf = typeof p.smoothed_confidence === 'number' ? p.smoothed_confidence : 0;
              const rawConf = typeof p.raw_confidence === 'number' ? p.raw_confidence : 0;

              const smoothedCandidate: 'focused' | 'distracted' | null =
                smoothedConf >= FAST_UI_CONFIDENCE_THRESHOLD
                  ? (p.smoothed_label === 'screen' ? 'focused' : 'distracted')
                  : null;
              const rawCandidate: 'focused' | 'distracted' | null =
                rawConf >= FAST_UI_CONFIDENCE_THRESHOLD
                  ? (p.raw_label === 'screen' ? 'focused' : 'distracted')
                  : null;
              const uiCandidate: 'focused' | 'distracted' | null = smoothedCandidate ?? rawCandidate;

              const now = Date.now();
              if (uiCandidate !== candidateFocusStateRef.current) {
                candidateFocusStateRef.current = uiCandidate;
                candidateSinceMsRef.current = uiCandidate ? now : null;
              } else if (uiCandidate && candidateSinceMsRef.current != null) {
                const elapsed = now - candidateSinceMsRef.current;
                const requiredMs =
                  uiCandidate === 'distracted'
                    ? FAST_UI_DISTRACT_HOLD_MS
                    : FAST_UI_REFOCUS_HOLD_MS;
                if (elapsed >= requiredMs) {
                  applyUiFocusState(uiCandidate);
                  announceFocusState(uiCandidate);
                  candidateFocusStateRef.current = null;
                  candidateSinceMsRef.current = null;
                }
              }
            },
          });
          await remoteTracker.start();
          tracker = remoteTracker;
          setLastPose(remoteTracker.getLastPrediction());
        } catch (remoteError) {
          console.warn('Could not start server inference tracker. Falling back to local webcam tracker.', remoteError);
          let lastFocused: boolean | null = null;
          const acquired = await acquireWebcamStream();
          trackerReleaseRef.current = acquired.release;
          const fallbackTracker = new LaptopFocusTracker({
            stream: acquired.stream,
            onSample: ({ isFocused }) => {
              const nextState: 'focused' | 'distracted' = isFocused ? 'focused' : 'distracted';
              if (lastFocused == null) {
                lastFocused = isFocused;
                applyUiFocusState(nextState);
                return;
              }
              if (lastFocused !== isFocused) {
                applyUiFocusState(nextState);
                announceFocusState(nextState);
              }
              lastFocused = isFocused;
            },
          });
          await fallbackTracker.start();
          tracker = fallbackTracker;
          trackerSource = 'laptop_webcam';
          label = 'Laptop webcam (fallback)';
          setLastPose(null);
        }

        localFocusTrackerRef.current = tracker;
        trackerSourceRef.current = trackerSource;
        setActiveTrackerLabel(label);
        setIsLocalTrackerRunning(true);
        showToast(`Focus tracking started (${res.focusSessionId.slice(0, 6)}...) via ${label}.`, 'success');
      } catch (trackerError) {
        localFocusTrackerRef.current = null;
        trackerSourceRef.current = null;
        setActiveTrackerLabel(null);
        setLastPose(null);
        setIsLocalTrackerRunning(false);
        console.warn('Could not start any focus tracker:', trackerError);
        showToast('Focus started, but tracking is unavailable on this device.', 'warning');
      }
    } catch (error) {
      console.error('Error starting focus session:', error);
      showToast("Error starting focus tracking", 'warning');
    } finally {
      setFocusBusy(false);
      setPendingFocusStart(null);
    }
  };

  const handleStopFocus = async (): Promise<FocusStopResult | null> => {
    if (!user || !activeFocusSession) return null;
    setFocusBusy(true);
    const tracker = localFocusTrackerRef.current;
    const trackerSource = trackerSourceRef.current;
    localFocusTrackerRef.current = null;
    trackerSourceRef.current = null;
    const releaseTrackerStream = trackerReleaseRef.current;
    trackerReleaseRef.current = null;
    setActiveTrackerLabel(null);
    setLastPose(null);
    setIsLocalTrackerRunning(false);
    let sessionStopped = false;
    const uiDistractions = uiDistractionsRef.current;
    const firstDriftSec = firstDriftOffsetSecRef.current;
    let stopResult: FocusStopResult = {
      ok: false,
      focusSessionId: activeFocusSession.id,
      source: trackerSource ?? null,
      summary: null,
      uiDistractions,
      firstDriftOffsetSec: firstDriftSec,
    };
    try {
      await stopFocusSession({
        userId: user.uid,
        focusSessionId: activeFocusSession.id,
      });
      sessionStopped = true;

      if (tracker) {
        const uiDistractions = uiDistractionsRef.current;
        const summary = await tracker.stop();
        const finalSummary =
          trackerSource === 'ml_inference_api'
            ? {
                ...summary,
                distractions: uiDistractions,
                distractionsSource: 'ui_fast',
              }
            : summary;
        await setDoc(
          doc(db, 'focusSummaries', activeFocusSession.id),
          {
            focusSessionId: activeFocusSession.id,
            userId: user.uid,
            source: trackerSource ?? 'laptop_webcam',
            courseId: activeFocusSession.courseId || null,
            sessionId: activeFocusSession.sessionId || null,
            mode: activeFocusSession.mode || null,
            chatId: activeFocusSession.chatId || null,
            examSimulationId: activeFocusSession.examSimulationId || null,
            firstDriftOffsetSec: firstDriftSec,
            createdAt: serverTimestamp(),
            ...finalSummary,
          },
          { merge: true }
        );
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
        void applyGamificationForFocusSession({
          userId: user.uid,
          focusSessionId: activeFocusSession.id,
          timezone,
        }).catch((gamificationError) => {
          console.warn('Gamification update failed for focus session:', gamificationError);
        });
        const sourceText = trackerSource === 'ml_inference_api' ? 'server ML inference' : 'laptop webcam';
        showToast(`Focus tracking stopped. Summary saved from ${sourceText}.`, 'success');
        stopResult = {
          ok: true,
          focusSessionId: activeFocusSession.id,
          source: trackerSource ?? null,
          summary: finalSummary,
          uiDistractions,
          firstDriftOffsetSec: firstDriftSec,
        };
        resetUiDistractionCounter();
      } else {
        showToast('Focus tracking stopped. No tracking summary was captured.', 'info');
        stopResult = {
          ok: true,
          focusSessionId: activeFocusSession.id,
          source: trackerSource ?? null,
          summary: null,
          uiDistractions,
          firstDriftOffsetSec: firstDriftSec,
        };
        resetUiDistractionCounter();
      }
      releaseTrackerStream?.();
      return stopResult;
    } catch (error) {
      if (tracker && !sessionStopped) {
        localFocusTrackerRef.current = tracker;
        trackerSourceRef.current = trackerSource;
        trackerReleaseRef.current = releaseTrackerStream ?? null;
        if (trackerSource === 'ml_inference_api') {
          setActiveTrackerLabel('Server ML inference');
        } else if (trackerSource === 'laptop_webcam') {
          setActiveTrackerLabel('Laptop webcam (fallback)');
        }
        setIsLocalTrackerRunning(true);
      }
      console.error('Error stopping focus session:', error);
      showToast("Error stopping focus tracking", 'warning');
      return stopResult;
    } finally {
      setFocusBusy(false);
    }
  };

  const cleanupOnSignOut = async () => {
    const tracker = localFocusTrackerRef.current;
    localFocusTrackerRef.current = null;
    trackerSourceRef.current = null;
    if (trackerReleaseRef.current) {
      trackerReleaseRef.current();
      trackerReleaseRef.current = null;
    }
    if (tracker) {
      try {
        await tracker.stop();
      } catch (trackerError) {
        console.warn('Error stopping tracker during sign out:', trackerError);
      }
    }
    setIsLocalTrackerRunning(false);
    setActiveTrackerLabel(null);
    setLastPose(null);
    setFocusElapsedMs(0);
    resetUiDistractionCounter();
    setActiveFocusSession(null);
    setPendingFocusStart(null);
    setShowCalibrationModal(false);
  };

  return {
    activeFocusSession,
    focusBusy,
    isLocalTrackerRunning,
    activeTrackerLabel,
    lastPose,
    focusElapsedMs,
    currentFocusState,
    studyDistractions,
    firstDriftOffsetSec,
    showCalibrationModal,
    handleStartFocus,
    startFocusAfterCalibration,
    handleStopFocus,
    hideCalibrationModal,
    cancelCalibration,
    cleanupOnSignOut,
  };
}
