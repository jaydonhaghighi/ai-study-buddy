import { useState, useEffect, useRef } from 'react';
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
import { getAIResponse } from '../services/genkit-service';
import { acquireWebcamStream } from '../services/webcam-manager';
import WebcamCalibrationPreview from './WebcamCalibrationPreview';
import FocusDashboard from './FocusDashboard';
import ChatMainHeader from './chat/ChatMainHeader';
import ChatMessageList from './chat/ChatMessageList';
import ChatInput from './chat/ChatInput';
import ChatSidebar from './chat/ChatSidebar';
import { useChatAutoScroll } from './chat/useChatAutoScroll';
import { useFocusTracking } from './chat/useFocusTracking';
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

  // Always-on camera preview (local webapp usage)
  const [cameraPreviewEnabled, setCameraPreviewEnabled] = useState(() => {
    // default ON for local usage
    return localStorage.getItem('cameraPreviewEnabled') !== '0';
  });
  const [cameraPreviewAfterCalibration, setCameraPreviewAfterCalibration] = useState(false);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewReleaseRef = useRef<null | (() => void)>(null);
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

  const {
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
  } = useFocusTracking({
    user,
    expandedCourseId,
    expandedSessionId,
    showToast,
    playFocusTransitionSound,
  });

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

  const {
    messagesEndRef,
    messagesContainerRef,
    handleMessagesScroll,
    enableAutoScroll,
  } = useChatAutoScroll({
    selectedChatId,
    mainView,
    messages,
    loading,
  });

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
      // Create the chat doc directly for instant UI responsiveness.
      // The Genkit session is initialized lazily on first message.
      const chatRef = doc(collection(db, 'chats'));
      await setDoc(chatRef, {
        id: chatRef.id,
        name: 'New Chat',
        userId: user.uid,
        courseId: expandedCourseId,
        sessionId: expandedSessionId,
        createdAt: serverTimestamp(),
        lastMessageAt: serverTimestamp(),
      });
      setSelectedChatId(chatRef.id);
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
    enableAutoScroll();

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
      await cleanupOnSignOut();
      await signOut(auth);
      setCourses([]);
      setSessions([]);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
    } catch (error) {
      console.error("Sign out error:", error);
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
  const visibleMessages = messages.filter((m) => {
    // Avoid rendering empty AI bubbles (e.g. streaming placeholder before first chunk).
    if (m.isAI && (!m.text || m.text.trim().length === 0)) return false;
    return true;
  });
  const hasStreamingAiText = messages.some(
    (m) => m.isAI && m.id.startsWith('temp-') && !!m.text && m.text.trim().length > 0
  );

  return (
    <div className={`chat-container ${cameraPreviewEnabled && mainView === 'chat' ? 'preview-sidebar-open' : ''}`}>
      <ChatSidebar
        courses={courses}
        sessions={sessions}
        chats={chats}
        expandedCourseId={expandedCourseId}
        expandedSessionId={expandedSessionId}
        selectedChatId={selectedChatId}
        isCreatingCourse={isCreatingCourse}
        newCourseName={newCourseName}
        isCreatingSession={isCreatingSession}
        newSessionName={newSessionName}
        editingChatId={editingChatId}
        editChatName={editChatName}
        onSetCreatingCourse={setIsCreatingCourse}
        onSetNewCourseName={setNewCourseName}
        onCreateCourse={handleCreateCourse}
        onToggleCourse={(courseId) => setExpandedCourseId(expandedCourseId === courseId ? null : courseId)}
        onSetCreatingSession={setIsCreatingSession}
        onSetNewSessionName={setNewSessionName}
        onCreateSession={handleCreateSession}
        onToggleSession={(sessionId) => setExpandedSessionId(expandedSessionId === sessionId ? null : sessionId)}
        onCreateChat={handleCreateChat}
        onSelectChat={setSelectedChatId}
        onSetEditingChatId={setEditingChatId}
        onSetEditChatName={setEditChatName}
        onUpdateChatName={handleUpdateChatName}
        onDeleteChat={handleDeleteChat}
      />

      {/* Main Area */}
      <div className="chat-main">
        <ChatMainHeader
          mainView={mainView}
          currentChatName={currentChat?.name}
          focusBusy={focusBusy}
          isFocusActive={!!activeFocusSession}
          activeTrackerLabel={activeTrackerLabel}
          focusElapsedMs={focusElapsedMs}
          lastPose={lastPose}
          settingsOpen={settingsOpen}
          cameraPreviewEnabled={cameraPreviewEnabled}
          settingsRef={settingsRef}
          settingsIconSrc={settingsIcon}
          onToggleMainView={() => setMainView(mainView === 'dashboard' ? 'chat' : 'dashboard')}
          onStartFocus={handleStartFocus}
          onStopFocus={handleStopFocus}
          onToggleSettings={() => setSettingsOpen((v) => !v)}
          onToggleCameraPreview={() => setCameraPreviewEnabled((v) => !v)}
          onSignOut={handleSignOut}
          formatDuration={formatDuration}
        />

        {mainView === 'dashboard' ? (
          <FocusDashboard userId={user.uid} />
        ) : (
          <>
            <ChatMessageList
              selectedChatId={selectedChatId}
              visibleMessages={visibleMessages}
              loading={loading}
              hasStreamingAiText={hasStreamingAiText}
              messagesContainerRef={messagesContainerRef}
              messagesEndRef={messagesEndRef}
              onMessagesScroll={handleMessagesScroll}
            />

            <ChatInput
              selectedChatId={selectedChatId}
              input={input}
              loading={loading}
              onInputChange={setInput}
              onSubmit={handleSend}
            />
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
              cancelCalibration();
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
                  cancelCalibration();
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
                  cancelCalibration();
                }}
                onAlignedStable={() => {
                  hideCalibrationModal();
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
