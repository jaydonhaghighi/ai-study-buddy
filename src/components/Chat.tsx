import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css';
import { db, auth } from '../firebase-config';
import { User, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from 'firebase/auth';
import { 
  collection, 
  addDoc, 
  setDoc,
  query, 
  where,
  orderBy,
  serverTimestamp,
  Timestamp,
  onSnapshot,
  deleteDoc,
  updateDoc,
  doc
} from 'firebase/firestore';
import { getAIResponse, createGenkitChat } from '../services/genkit-service';
import { startFocusSession, stopFocusSession } from '../services/focus-service';
import { LaptopFocusTracker } from '../services/laptop-focus-tracker';
import { InferenceFocusTracker, type InferencePredictionPayload } from '../services/inference-focus-tracker';
import { acquireWebcamStream } from '../services/webcam-manager';
import WebcamCalibrationPreview from './WebcamCalibrationPreview';
import FocusDashboard from './FocusDashboard';
import settingsIcon from '../public/settings.svg';
import './Chat.css';

interface Message {
  id: string;
  text: string;
  userId: string;
  userName: string;
  sessionId: string; // This corresponds to chatId now
  createdAt: Date | null;
  isAI?: boolean;
  model?: string;
}

interface Course {
  id: string;
  name: string;
  userId: string;
}

interface Session {
  id: string;
  courseId: string;
  name: string;
  userId: string;
  createdAt: Date | null;
}

interface ChatSession {
  id: string;
  sessionId: string;
  courseId: string;
  name: string;
  userId: string;
  createdAt: Date | null;
  lastMessageAt: Date | null;
}

interface FocusSession {
  id: string;
  userId: string;
  status: string;
  courseId?: string | null;
  sessionId?: string | null;
  startedAt?: Date | null;
}

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

type TrackerSource = 'ml_inference_api' | 'laptop_webcam';

interface ChatProps {
  user: User | null;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

export default function Chat({ user }: ChatProps) {
  // Navigation State
  const [courses, setCourses] = useState<Course[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [mainView, setMainView] = useState<'chat' | 'dashboard'>('chat');
  
  const [expandedCourseId, setExpandedCourseId] = useState<string | null>(null);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);

  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Auth State
  const [showAuth, setShowAuth] = useState(false);
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  // UI State
  const [isCreatingCourse, setIsCreatingCourse] = useState(false);
  const [newCourseName, setNewCourseName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editChatName, setEditChatName] = useState('');
  
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastVariant, setToastVariant] = useState<'success' | 'warning' | 'info'>('success');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const announcedFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const candidateSinceMsRef = useRef<number | null>(null);
  const lastAnnounceMsRef = useRef<number>(0);
  const uiFocusStateRef = useRef<'focused' | 'distracted' | null>(null);
  const uiDistractionsRef = useRef<number>(0);

  const showToast = (message: string, variant: 'success' | 'warning' | 'info' = 'success') => {
    setToastVariant(variant);
    setToastMessage(message);
  };

  const playBeep = (frequencyHz: number, durationMs: number, gain: number = 0.06) => {
    try {
      const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextCtor) return;
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContextCtor();
      }
      const ctx = audioCtxRef.current;
      // best-effort resume; may still be blocked until user gesture (Start Focus counts)
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = frequencyHz;
      g.gain.value = gain;
      osc.connect(g);
      g.connect(ctx.destination);
      const now = ctx.currentTime;
      osc.start(now);
      osc.stop(now + durationMs / 1000);
    } catch {
      // ignore audio failures (permissions/autoplay policy)
    }
  };

  const playFocusTransitionSound = (nextState: 'focused' | 'distracted') => {
    if (nextState === 'distracted') {
      playBeep(220, 140);
      window.setTimeout(() => playBeep(220, 140), 170);
    } else {
      playBeep(660, 180);
    }
  };

  const announceFocusState = (nextState: 'focused' | 'distracted', mode: 'fast' | 'transitioned') => {
    // Prevent duplicate announcements
    if (announcedFocusStateRef.current === nextState) return;

    // Rate-limit to avoid spam when jittering
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

  // Always-on camera preview (local webapp usage)
  const [cameraPreviewEnabled, setCameraPreviewEnabled] = useState(() => {
    // default ON for local usage
    return localStorage.getItem('cameraPreviewEnabled') !== '0';
  });
  const [cameraPreviewAfterCalibration, setCameraPreviewAfterCalibration] = useState(false);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewReleaseRef = useRef<null | (() => void)>(null);
  const trackerReleaseRef = useRef<null | (() => void)>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem('cameraPreviewEnabled', cameraPreviewEnabled ? '1' : '0');
  }, [cameraPreviewEnabled]);

  // Shared camera preview: use a single MediaStream for preview (and potentially tracking).
  useEffect(() => {
    const shouldShowPreview = cameraPreviewEnabled && cameraPreviewAfterCalibration && mainView === 'chat';
    let cancelled = false;

    const stopPreview = () => {
      if (previewReleaseRef.current) {
        previewReleaseRef.current();
        previewReleaseRef.current = null;
      }
      if (previewVideoRef.current) {
        previewVideoRef.current.pause();
        previewVideoRef.current.srcObject = null;
      }
    };

    if (!shouldShowPreview) {
      stopPreview();
      return;
    }

    void (async () => {
      try {
        setPreviewError(null);
        // Acquire shared stream for preview.
        const { stream, release } = await acquireWebcamStream();
        if (cancelled) {
          release();
          return;
        }
        previewReleaseRef.current = release;
        const el = previewVideoRef.current;
        if (!el) return;
        el.srcObject = stream;
        await el.play();
      } catch (e: any) {
        setPreviewError(e?.message || 'Could not start camera preview');
      }
    })();

    return () => {
      cancelled = true;
      stopPreview();
    };
  }, [cameraPreviewEnabled, cameraPreviewAfterCalibration, mainView]);

  // Focus Tracking State
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
  const focusStartLocalMsRef = useRef<number | null>(null);

  // Avoid camera conflicts: pause shared preview while calibration modal is open.
  useEffect(() => {
    if (!showCalibrationModal) return;
    if (previewReleaseRef.current) {
      previewReleaseRef.current();
      previewReleaseRef.current = null;
    }
    if (previewVideoRef.current) {
      previewVideoRef.current.pause();
      previewVideoRef.current.srcObject = null;
    }
  }, [showCalibrationModal]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!settingsOpen) return;
    const onMouseDown = (e: MouseEvent) => {
      const el = settingsRef.current;
      if (!el) return;
      if (e.target instanceof Node && !el.contains(e.target)) {
        setSettingsOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSettingsOpen(false);
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [settingsOpen]);

  useEffect(() => {
    if (toastMessage) {
      const timer = setTimeout(() => setToastMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [toastMessage]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 1. Fetch Courses
  useEffect(() => {
    if (!user) {
      setCourses([]);
      return;
    }

    const q = query(
      collection(db, 'courses'),
      where('userId', '==', user.uid),
      orderBy('name', 'asc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      setCourses(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as Course)));
    });

    return () => unsubscribe();
  }, [user]);

  // Active focus session (assume at most 1 active per user for MVP)
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
      if (previewReleaseRef.current) {
        previewReleaseRef.current();
        previewReleaseRef.current = null;
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

  // 2. Fetch Sessions (when a course is expanded)
  useEffect(() => {
    if (!user || !expandedCourseId) {
      setSessions([]);
      return;
    }

    const q = query(
      collection(db, 'sessions'),
      where('userId', '==', user.uid),
      where('courseId', '==', expandedCourseId),
      orderBy('createdAt', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      setSessions(snapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null
        } as Session;
      }));
    });

    return () => unsubscribe();
  }, [user, expandedCourseId]);

  // 3. Fetch Chats (when a session is expanded)
  useEffect(() => {
    if (!user || !expandedSessionId) {
      setChats([]);
      return;
    }

    const q = query(
      collection(db, 'chats'),
      where('userId', '==', user.uid),
      where('sessionId', '==', expandedSessionId),
      orderBy('lastMessageAt', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      setChats(snapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null,
          lastMessageAt: data.lastMessageAt instanceof Timestamp ? data.lastMessageAt.toDate() : null
        } as ChatSession;
      }));
    });

    return () => unsubscribe();
  }, [user, expandedSessionId]);

  // 4. Fetch Messages (when a chat is selected)
  useEffect(() => {
    if (mainView !== 'chat') {
      setMessages([]);
      return;
    }
    if (!user || !selectedChatId) {
      setMessages([]);
      return;
    }

    const q = query(
      collection(db, 'messages'),
      where('sessionId', '==', selectedChatId),
      where('userId', '==', user.uid),
      orderBy('createdAt', 'asc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const msgs: Message[] = [];
      snapshot.forEach((doc) => {
        const data = doc.data();
        let messageText = data.text;
        if (typeof messageText === 'object' && messageText !== null) {
          messageText = messageText.text || messageText.content || JSON.stringify(messageText);
        }
        msgs.push({
          id: doc.id,
          text: String(messageText || ''),
          userId: data.userId,
          userName: data.userName || 'User',
          sessionId: data.sessionId,
          createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null,
          isAI: data.isAI || false,
          model: data.model
        });
      });
      setMessages(msgs);
    });

    return () => unsubscribe();
  }, [user, selectedChatId, mainView]);

  // Actions
  const handleCreateCourse = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !newCourseName.trim()) return;
    try {
      await addDoc(collection(db, 'courses'), {
        userId: user.uid,
        name: newCourseName.trim(),
        createdAt: serverTimestamp()
      });
      setNewCourseName('');
      setIsCreatingCourse(false);
    } catch (error) {
      console.error("Error creating course:", error);
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !expandedCourseId || !newSessionName.trim()) return;
    try {
      const docRef = await addDoc(collection(db, 'sessions'), {
        userId: user.uid,
        courseId: expandedCourseId,
        name: newSessionName.trim(),
        createdAt: serverTimestamp(),
        focusScore: 0
      });
      setNewSessionName('');
      setIsCreatingSession(false);
      setExpandedSessionId(docRef.id); // Auto expand
    } catch (error) {
      console.error("Error creating session:", error);
    }
  };

  const handleCreateChat = async () => {
    if (!user || !expandedCourseId || !expandedSessionId) return;
    try {
      const chat = await createGenkitChat(user.uid, expandedCourseId, expandedSessionId, 'New Chat');
      setSelectedChatId(chat.chatId);
    } catch (error) {
      console.error("Error creating chat:", error);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user || !selectedChatId) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    try {
      // Save user message
      await addDoc(collection(db, 'messages'), {
        text: userMessage,
        userId: user.uid,
        userName: user.email?.split('@')[0] || 'User',
        sessionId: selectedChatId,
        createdAt: serverTimestamp(),
        isAI: false
      });

      // Update chat timestamp
      await updateDoc(doc(db, 'chats', selectedChatId), {
        lastMessageAt: serverTimestamp()
      });

      // Streaming placeholder
      const tempId = `temp-${Date.now()}`;
      setMessages(prev => [...prev, {
        id: tempId,
        text: '',
        userId: user.uid,
        userName: 'AI Study Buddy',
        sessionId: selectedChatId,
        createdAt: new Date(),
        isAI: true
      }]);

      // Call AI
      let streamingText = '';
      const response = await getAIResponse(userMessage, selectedChatId, user.uid, (chunk) => {
        streamingText += chunk;
        setMessages(prev => prev.map(m => m.id === tempId ? { ...m, text: streamingText } : m));
      });

      // Remove placeholder and add real message
      setMessages(prev => prev.filter(m => m.id !== tempId));
      await addDoc(collection(db, 'messages'), {
        text: response.text,
        userId: user.uid,
        userName: 'AI Study Buddy',
        sessionId: selectedChatId,
        createdAt: serverTimestamp(),
        isAI: true,
        model: response.model
      });

      // Update chat timestamp again
      await updateDoc(doc(db, 'chats', selectedChatId), {
        lastMessageAt: serverTimestamp()
      });

    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleUpdateChatName = async (chatId: string, newName: string) => {
    if (!user || !newName.trim()) return;
    try {
      await updateDoc(doc(db, 'chats', chatId), { name: newName.trim() });
      setEditingChatId(null);
    } catch (error) {
      console.error('Error updating chat name:', error);
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      await deleteDoc(doc(db, 'chats', chatId));
      if (selectedChatId === chatId) setSelectedChatId(null);
      showToast("Chat deleted", "success");
    } catch (error) {
      console.error('Error deleting chat:', error);
      showToast("Error deleting chat", "warning");
    }
  };

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, authEmail, authPassword);
      } else {
        await signInWithEmailAndPassword(auth, authEmail, authPassword);
      }
      setAuthEmail('');
      setAuthPassword('');
      setShowAuth(false);
    } catch (err: any) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignOut = async () => {
    try {
      setSettingsOpen(false);
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
      await signOut(auth);
      setCourses([]);
      setSessions([]);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
      setActiveFocusSession(null);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  const handleStartFocus = async () => {
    if (!user) return;
    // Optional: link focus session to the currently expanded course/session (chapter)
    const courseId = expandedCourseId || undefined;
    const sessionId = expandedSessionId || undefined;

    // Calibration gate: do not start focus until calibration completes
    setPendingFocusStart({ courseId, sessionId });
    setShowCalibrationModal(true);
  };

  const startFocusAfterCalibration = async () => {
    if (!user) return;
    if (!pendingFocusStart) return;

    setFocusBusy(true);
    try {
      resetUiDistractionCounter();
      const res = await startFocusSession({
        userId: user.uid,
        courseId: pendingFocusStart.courseId,
        sessionId: pendingFocusStart.sessionId,
      });
      focusStartLocalMsRef.current = Date.now();

      // Start server inference tracker first; fallback to local tracker if server is unavailable.
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
              // 5 FPS updates are fine, but avoid extra renders if nothing changed
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
                if (p.state === 'distracted') {
                  announceFocusState('distracted', 'transitioned');
                } else if (p.state === 'focused') {
                  announceFocusState('focused', 'transitioned');
                }
              }

              // Fast UI-only detector (does NOT affect logged summaries):
              // If the model's smoothed label is away/screen with sufficient confidence,
              // notify after a short debounce so quick in/out still registers.
              const conf = typeof p.smoothed_confidence === 'number' ? p.smoothed_confidence : 0;
              const uiCandidate: 'focused' | 'distracted' | null =
                conf >= 0.55 ? (p.smoothed_label === 'screen' ? 'focused' : 'distracted') : null;

              const now = Date.now();
              if (uiCandidate !== candidateFocusStateRef.current) {
                candidateFocusStateRef.current = uiCandidate;
                candidateSinceMsRef.current = uiCandidate ? now : null;
              } else if (uiCandidate && candidateSinceMsRef.current != null) {
                const elapsed = now - candidateSinceMsRef.current;
                const requiredMs = uiCandidate === 'distracted' ? 600 : 300;
                if (elapsed >= requiredMs) {
                  // Use the fast UI detector for the distraction counter.
                  applyUiFocusState(uiCandidate);
                  announceFocusState(uiCandidate, 'fast');
                  // Reset so it doesn't re-announce continuously.
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
        // Capture before any possible async resets/races.
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

  if (!user) {
    // Auth View
    return (
      <div className="chat-container chat-container-auth">
        <div className="chat-header chat-header-auth">
          <div className="chat-header-inner">
            <div className="chat-header-left">
              <h2 className="chat-header-title">AI Study Buddy</h2>
              <p className="chat-header-subtitle">Sign in to start learning.</p>
            </div>
          </div>
        </div>
        {!showAuth ? (
          <div className="chat-placeholder">
            <div className="auth-prompt">
              <p className="auth-prompt-title">Organize your learning with Courses & Sessions</p>
              <p className="auth-prompt-subtitle">Create a focused workspace for every subject, and keep your study chats in context.</p>
              <button className="auth-toggle-button" onClick={() => setShowAuth(true)}>Sign In / Sign Up</button>
            </div>
          </div>
        ) : (
          <div className="chat-auth-form">
            <form onSubmit={handleAuth}>
              <h3>{isSignUp ? 'Create Account' : 'Sign In'}</h3>
              {authError && <div className="auth-error">{authError}</div>}
              <input type="email" placeholder="Email" value={authEmail} onChange={e => setAuthEmail(e.target.value)} className="auth-input" required />
              <input type="password" placeholder="Password" value={authPassword} onChange={e => setAuthPassword(e.target.value)} className="auth-input" required />
              <button type="submit" className="auth-submit-button" disabled={authLoading}>{authLoading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}</button>
              <button type="button" className="auth-switch-button" onClick={() => setIsSignUp(!isSignUp)}>
                {isSignUp ? 'Have account? Sign In' : 'No account? Sign Up'}
              </button>
              <button type="button" className="auth-cancel-button" onClick={() => setShowAuth(false)}>Cancel</button>
            </form>
          </div>
        )}
      </div>
    );
  }

  const currentChat = chats.find(c => c.id === selectedChatId);

  return (
    <div className={`chat-container ${cameraPreviewEnabled && mainView === 'chat' ? 'preview-sidebar-open' : ''}`}>
      {/* Sidebar */}
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <h3>My Courses</h3>
          <button onClick={() => setIsCreatingCourse(true)} className="add-button" title="Add Course">+</button>
        </div>

        {isCreatingCourse && (
          <form onSubmit={handleCreateCourse} className="create-form">
            <input 
              autoFocus
              value={newCourseName}
              onChange={e => setNewCourseName(e.target.value)}
              placeholder="Course Name"
              onBlur={() => setIsCreatingCourse(false)}
            />
          </form>
        )}

        <div className="sidebar-content">
          {courses.map(course => (
            <div key={course.id} className="course-group">
              <div 
                className={`course-item ${expandedCourseId === course.id ? 'expanded' : ''}`}
                onClick={() => setExpandedCourseId(expandedCourseId === course.id ? null : course.id)}
              >
                <span className="name">{course.name}</span>
                <span className="dropdown-arrow">›</span>
              </div>

              {expandedCourseId === course.id && (
                <div className="session-list">
                  <div className="session-header">
                    <small>Sessions</small>
                    <button onClick={() => setIsCreatingSession(true)} className="add-button-small">+</button>
                  </div>
                  
                  {isCreatingSession && (
                    <form onSubmit={handleCreateSession} className="create-form-small">
                      <input 
                        autoFocus
                        value={newSessionName}
                        onChange={e => setNewSessionName(e.target.value)}
                        placeholder="Session Name"
                        onBlur={() => setIsCreatingSession(false)}
                      />
                    </form>
                  )}

                  {sessions.map(session => (
                    <div key={session.id} className="session-group">
                      <div 
                        className={`session-item ${expandedSessionId === session.id ? 'expanded' : ''}`}
                        onClick={() => setExpandedSessionId(expandedSessionId === session.id ? null : session.id)}
                      >
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            viewBox="0 0 24 24" 
                            className="folder-icon"
                          >
                            <path d="M19,3H12.472a1.019,1.019,0,0,1-.447-.1L8.869,1.316A3.014,3.014,0,0,0,7.528,1H5A5.006,5.006,0,0,0,0,6V18a5.006,5.006,0,0,0,5,5H19a5.006,5.006,0,0,0,5-5V8A5.006,5.006,0,0,0,19,3ZM5,3H7.528a1.019,1.019,0,0,1,.447.1l3.156,1.579A3.014,3.014,0,0,0,12.472,5H19a3,3,0,0,1,2.779,1.882L2,6.994V6A3,3,0,0,1,5,3ZM19,21H5a3,3,0,0,1-3-3V8.994l20-.113V18A3,3,0,0,1,19,21Z"/>
                          </svg>
                          <span className="name">{session.name}</span>
                        </div>
                        <span className="dropdown-arrow">›</span>
                      </div>

                      {expandedSessionId === session.id && (
                        <div className="chat-list">
                          <button onClick={handleCreateChat} className="new-chat-button-small">+ New Chat</button>
                          {chats.map(chat => (
                            <div 
                              key={chat.id} 
                              className={`chat-item ${selectedChatId === chat.id ? 'active' : ''}`}
                              onClick={() => setSelectedChatId(chat.id)}
                            >
                              {editingChatId === chat.id ? (
                                <input
                                  value={editChatName}
                                  onChange={e => setEditChatName(e.target.value)}
                                  onBlur={() => handleUpdateChatName(chat.id, editChatName)}
                                  onKeyDown={e => {
                                    if(e.key === 'Enter') handleUpdateChatName(chat.id, editChatName);
                                    if(e.key === 'Escape') setEditingChatId(null);
                                  }}
                                  autoFocus
                                  onClick={e => e.stopPropagation()}
                                  className="chat-name-input"
                                />
                              ) : (
                                <>
                                  <span className="chat-name" onDoubleClick={(e) => {
                                    e.stopPropagation();
                                    setEditingChatId(chat.id);
                                    setEditChatName(chat.name);
                                  }}>{chat.name}</span>
                                  <div className="chat-actions">
                                    <button onClick={(e) => {
                                      e.stopPropagation();
                                      setEditingChatId(chat.id);
                                      setEditChatName(chat.name);
                                    }} className="action-btn">✎</button>
                                    <button onClick={(e) => {
                                      e.stopPropagation();
                                      handleDeleteChat(chat.id);
                                    }} className="action-btn">×</button>
                                  </div>
                                </>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                  {sessions.length === 0 && !isCreatingSession && (
                    <div className="empty-state-small">No sessions yet</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Main Area */}
      <div className="chat-main">
        <div className="chat-header">
          <div className="chat-header-inner">
            <div className="chat-header-left">
              <h2 className="chat-header-title">
                {mainView === 'dashboard' ? 'Focus dashboard' : (currentChat?.name || 'Select a Chat')}
              </h2>
              <p className="chat-header-subtitle">
                {mainView === 'dashboard'
                  ? 'Visualize focus sessions captured from your webcam.'
                  : (currentChat ? 'AI Study Buddy' : 'Choose an existing chat or create a new one to get started')}
              </p>
            </div>

            <div className="chat-header-right">
              <div className="chat-header-controls">
                <div className="chat-header-row">
                  <button
                    onClick={() => setMainView(mainView === 'dashboard' ? 'chat' : 'dashboard')}
                    className="chat-header-btn"
                    disabled={focusBusy}
                    type="button"
                  >
                    {mainView === 'dashboard' ? 'Back to chat' : 'Dashboard'}
                  </button>

                  {!activeFocusSession ? (
                    <button
                      onClick={handleStartFocus}
                      className="chat-header-btn chat-header-btn-primary"
                      disabled={focusBusy}
                    >
                      Start Focus
                    </button>
                  ) : (
                    <button
                      onClick={handleStopFocus}
                      className="chat-header-btn chat-header-btn-danger"
                      disabled={focusBusy}
                    >
                      Stop Focus
                    </button>
                  )}

                  <div className="chat-settings" ref={settingsRef}>
                    <button
                      className="chat-header-btn chat-settings-btn"
                      type="button"
                      aria-label="Settings"
                      aria-haspopup="menu"
                      aria-expanded={settingsOpen}
                      onClick={() => setSettingsOpen((v) => !v)}
                      disabled={focusBusy}
                      title="Settings"
                    >
                      <img src={settingsIcon} alt="" className="chat-settings-icon" />
                    </button>

                    {settingsOpen && (
                      <div className="chat-settings-menu" role="menu" aria-label="Settings menu">
                        <button
                          type="button"
                          className="chat-settings-item"
                          role="menuitem"
                          onClick={() => setCameraPreviewEnabled((v) => !v)}
                          disabled={focusBusy}
                        >
                          Camera preview: {cameraPreviewEnabled ? 'On' : 'Off'}
                        </button>
                        <button
                          type="button"
                          className="chat-settings-item chat-settings-item-danger"
                          role="menuitem"
                          onClick={handleSignOut}
                        >
                          Sign out
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {activeFocusSession && (
                  <div className="chat-header-meta">
                    <span className="chat-header-pill">Focus active</span>
                    <span>{activeTrackerLabel ?? 'Webcam tracking'}</span>
                    <span>Duration: {formatDuration(focusElapsedMs)}</span>
                    {lastPose && (
                      <span>
                        Pose: {lastPose.smoothed_label} ({Math.round(lastPose.smoothed_confidence * 100)}%)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {mainView === 'dashboard' ? (
          <FocusDashboard userId={user.uid} />
        ) : (
          <>
            <div className="chat-messages" ref={messagesContainerRef}>
              {!selectedChatId ? (
                <div className="chat-welcome">
                  <div className="welcome-icon">💬</div>
                  <h3 className="welcome-title">Welcome to AI Study Buddy</h3>
                  <p className="welcome-message">To get started, choose an existing chat or create a new one from the sidebar</p>
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <div key={message.id} className={`message ${!message.isAI ? 'message-user' : 'message-ai'}`}>
                      <div className="message-content">
                        <div className="message-header">
                          <span className="message-name">{!message.isAI ? 'You' : (message.userName || 'AI Study Buddy')}</span>
                          {message.model && message.isAI && <span className="message-model">{message.model}</span>}
                        </div>
                        <div className="message-text">
                          {!message.isAI ? (
                            <div className="plain-text">{message.text}</div>
                          ) : (
                            <div className="markdown">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeHighlight]}
                                components={{
                                  a: ({ children, ...props }) => (
                                    <a {...props} target="_blank" rel="noopener noreferrer">
                                      {children}
                                    </a>
                                  ),
                                  code: ({ children, className, ...props }) => {
                                    const isBlock = !!className && className.includes('language-');
                                    return isBlock ? (
                                      <code className={className} {...props}>
                                        {children}
                                      </code>
                                    ) : (
                                      <code className="inline-code" {...props}>
                                        {children}
                                      </code>
                                    );
                                  },
                                }}
                              >
                                {message.text}
                              </ReactMarkdown>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  {loading && <div className="message message-ai"><div className="message-content"><div className="typing-indicator"><span></span><span></span><span></span></div></div></div>}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {selectedChatId && (
              <form className="chat-input-form" onSubmit={handleSend}>
                <div className="chat-input-wrapper">
                  <input
                    type="text"
                    className="chat-input"
                    placeholder="Type your message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={loading}
                  />
                  <button type="submit" className="chat-send-button" disabled={!input.trim() || loading}>
                    Send
                  </button>
                </div>
              </form>
            )}
          </>
        )}
      </div>

      {/* Preview Sidebar */}
      {cameraPreviewEnabled && mainView === 'chat' && (
        <div className="preview-sidebar" aria-label="Camera preview sidebar">
          <div className="preview-sidebar-header">
            <h3>Camera</h3>
          </div>

          <div className="preview-sidebar-body">
            {isLocalTrackerRunning ? (
              previewError ? (
                <div className="preview-sidebar-empty">
                  <p>Camera preview unavailable: {previewError}</p>
                </div>
              ) : (
                <video ref={previewVideoRef} className="preview-video" playsInline muted />
              )
            ) : !cameraPreviewAfterCalibration ? (
              <div className="preview-sidebar-empty">
                <p>
                  Start Focus to calibrate your webcam. Once calibration is complete, the live preview will stay here while you study.
                </p>
              </div>
            ) : (
              previewError ? (
                <div className="preview-sidebar-empty">
                  <p>Camera preview unavailable: {previewError}</p>
                </div>
              ) : (
                <video ref={previewVideoRef} className="preview-video" playsInline muted />
              )
            )}
          </div>
        </div>
      )}
      
      {toastMessage && (
        <div className={`toast-notification ${toastVariant === 'warning' ? 'toast-warning' : toastVariant === 'info' ? 'toast-info' : ''}`}>
          {toastMessage}
        </div>
      )}

      {showCalibrationModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Camera calibration"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) {
              setShowCalibrationModal(false);
              setPendingFocusStart(null);
            }
          }}
        >
          <div className="modal modal--wide">
            <div className="modal-header">
              <div>
                <h3 className="modal-title">Camera calibration</h3>
                <p className="modal-subtitle">Align your laptop webcam before focus tracking starts.</p>
              </div>
              <button
                className="modal-close"
                onClick={() => {
                  setShowCalibrationModal(false);
                  setPendingFocusStart(null);
                }}
                aria-label="Close"
                type="button"
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              <WebcamCalibrationPreview
                variant="embedded"
                autoStart
                mode="calibration"
                autoStopAfterAlignedSeconds={3}
                stopOnAlignedStable={false}
                onRequestClose={() => {
                  setShowCalibrationModal(false);
                  setPendingFocusStart(null);
                }}
                onAlignedStable={() => {
                  setShowCalibrationModal(false);
                  setCameraPreviewAfterCalibration(true);
                  startFocusAfterCalibration();
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
