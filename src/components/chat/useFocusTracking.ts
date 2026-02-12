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
import { LaptopFocusTracker } from '../../services/laptop-focus-tracker';
import {
  InferenceFocusTracker,
  type InferencePredictionPayload,
} from '../../services/inference-focus-tracker';
import { acquireWebcamStream } from '../../services/webcam-manager';

type ToastVariant = 'success' | 'warning' | 'info';
type TrackerSource = 'ml_inference_api' | 'laptop_webcam';

type FocusTrackerSummary = {
  startTs: number;
  endTs: number;
  focusedMs: number;
  distractedMs: number;
  distractions: number;
  focusPercent: number;
  attentionLabelCounts: Record<string, number>;
  [key: string]: unknown;
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
  startedAt?: Date | null;
};

const FAST_UI_CONFIDENCE_THRESHOLD = 0.45;
const FAST_UI_DISTRACT_HOLD_MS = 450;
const FAST_UI_REFOCUS_HOLD_MS = 250;

type UseFocusTrackingParams = {
  user: User | null;
  expandedCourseId: string | null;
  expandedSessionId: string | null;
  showToast: (message: string, variant?: ToastVariant) => void;
  playFocusTransitionSound: (nextState: 'focused' | 'distracted') => void;
};

export function useFocusTracking({
  user,
  expandedCourseId,
  expandedSessionId,
  showToast,
  playFocusTransitionSound,
}: UseFocusTrackingParams) {
  const [activeFocusSession, setActiveFocusSession] = useState<FocusSession | null>(null);
  const [focusBusy, setFocusBusy] = useState(false);
  const [isLocalTrackerRunning, setIsLocalTrackerRunning] = useState(false);
  const [activeTrackerLabel, setActiveTrackerLabel] = useState<string | null>(null);
  const [lastPose, setLastPose] = useState<InferencePredictionPayload | null>(null);
  const [focusElapsedMs, setFocusElapsedMs] = useState<number>(0);
  const [showCalibrationModal, setShowCalibrationModal] = useState(false);
  const [pendingFocusStart, setPendingFocusStart] = useState<{
    courseId?: string;
    sessionId?: string;
  } | null>(null);

  const localFocusTrackerRef = useRef<FocusTrackerRuntime | null>(null);
  const trackerSourceRef = useRef<TrackerSource | null>(null);
  const trackerReleaseRef = useRef<null | (() => void)>(null);
  const focusStartLocalMsRef = useRef<number | null>(null);

  const announcedFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateSinceMsRef = useRef<number | null>(null);
  const lastAnnounceMsRef = useRef<number>(0);
  const uiFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const uiDistractionsRef = useRef<number>(0);

  const announceFocusState = (nextState: 'focused' | 'distracted', mode: 'fast' | 'transitioned') => {
    if (announcedFocusStateRef.current === nextState) return;
    const now = Date.now();
    if (now - lastAnnounceMsRef.current < 800) return;

    announcedFocusStateRef.current = nextState;
    lastAnnounceMsRef.current = now;

    if (nextState === 'distracted') {
      showToast(mode === 'fast' ? 'You are distracted' : 'You are distracted', 'warning');
      playFocusTransitionSound('distracted');
    } else {
      showToast(mode === 'fast' ? "You're back in focus" : "You're back in focus", 'success');
      playFocusTransitionSound('focused');
    }
  };

  const resetUiDistractionCounter = () => {
    uiFocusStateRef.current = null;
    uiDistractionsRef.current = 0;
    candidateFocusStateRef.current = null;
    candidateSinceMsRef.current = null;
    announcedFocusStateRef.current = null;
    lastAnnounceMsRef.current = 0;
  };

  const applyUiFocusState = (nextState: 'focused' | 'distracted') => {
    const prev = uiFocusStateRef.current;
    if (prev === 'focused' && nextState === 'distracted') {
      uiDistractionsRef.current += 1;
    }
    uiFocusStateRef.current = nextState;
  };

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
        startedAt: data.startedAt instanceof Timestamp ? data.startedAt.toDate() : null,
      } as FocusSession);
    });

    return () => unsubscribe();
  }, [user]);

  useEffect(() => {
    return () => {
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
  }, [activeFocusSession?.id, activeFocusSession?.startedAt]);

  const handleStartFocus = () => {
    if (!user) return;
    const courseId = expandedCourseId || undefined;
    const sessionId = expandedSessionId || undefined;
    setPendingFocusStart({ courseId, sessionId });
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
      const res = await startFocusSession({
        userId: user.uid,
        courseId: pendingFocusStart.courseId,
        sessionId: pendingFocusStart.sessionId,
      });
      focusStartLocalMsRef.current = Date.now();

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
                  announceFocusState(transitionedState, 'transitioned');
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
                  announceFocusState(uiCandidate, 'fast');
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
              if (lastFocused == null) {
                lastFocused = isFocused;
                return;
              }
              if (lastFocused && !isFocused) {
                showToast('You are distracted', 'warning');
                playFocusTransitionSound('distracted');
              } else if (!lastFocused && isFocused) {
                showToast("You're back in focus", 'success');
                playFocusTransitionSound('focused');
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

  const handleStopFocus = async () => {
    if (!user || !activeFocusSession) return;
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
            createdAt: serverTimestamp(),
            ...finalSummary,
          },
          { merge: true }
        );
        const sourceText = trackerSource === 'ml_inference_api' ? 'server ML inference' : 'laptop webcam';
        showToast(`Focus tracking stopped. Summary saved from ${sourceText}.`, 'success');
        resetUiDistractionCounter();
      } else {
        showToast('Focus tracking stopped. No tracking summary was captured.', 'info');
        resetUiDistractionCounter();
      }
      releaseTrackerStream?.();
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
    showCalibrationModal,
    handleStartFocus,
    startFocusAfterCalibration,
    handleStopFocus,
    hideCalibrationModal,
    cancelCalibration,
    cleanupOnSignOut,
  };
}
