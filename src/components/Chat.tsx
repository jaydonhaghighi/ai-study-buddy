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
import { claimDevice, startFocusSession, stopFocusSession } from '../services/focus-service';
import PiCalibrationPreview from './PiCalibrationPreview';
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

interface Device {
  id: string;
  claimCode?: string;
  status?: string;
  pairedUserId?: string;
  activeFocusSessionId?: string | null;
}

interface FocusSession {
  id: string;
  userId: string;
  deviceId: string;
  status: string;
  courseId?: string | null;
  sessionId?: string | null;
}

interface ChatProps {
  user: User | null;
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
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  // Always-on camera preview (local webapp usage)
  const [cameraPreviewEnabled, setCameraPreviewEnabled] = useState(() => {
    // default ON for local usage
    return localStorage.getItem('cameraPreviewEnabled') !== '0';
  });
  const [cameraPreviewAfterCalibration, setCameraPreviewAfterCalibration] = useState(false);

  useEffect(() => {
    localStorage.setItem('cameraPreviewEnabled', cameraPreviewEnabled ? '1' : '0');
  }, [cameraPreviewEnabled]);

  // Device + Focus Tracking State
  const [claimCode, setClaimCode] = useState('');
  const [devices, setDevices] = useState<Device[]>([]);
  const [devicesLoaded, setDevicesLoaded] = useState(false);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [activeFocusSession, setActiveFocusSession] = useState<FocusSession | null>(null);
  const [focusBusy, setFocusBusy] = useState(false);
  const [showPairModal, setShowPairModal] = useState(false);
  const [showCalibrationModal, setShowCalibrationModal] = useState(false);
  const hasAutoOpenedPairModal = useRef(false);
  const userOverrodeDeviceSelectionRef = useRef(false);
  const [pendingFocusStart, setPendingFocusStart] = useState<{
    deviceId: string;
    courseId?: string;
    sessionId?: string;
  } | null>(null);

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

  // Devices (paired to user)
  useEffect(() => {
    if (!user) {
      setDevices([]);
      setDevicesLoaded(false);
      setSelectedDeviceId('');
      userOverrodeDeviceSelectionRef.current = false;
      return;
    }

    setDevicesLoaded(false);
    const q = query(
      collection(db, 'devices'),
      where('pairedUserId', '==', user.uid)
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const ds = snapshot.docs.map(d => ({ id: d.id, ...d.data() } as any));
      // Sort newest-first by pairedAt (fallback updatedAt, then createdAt).
      const sorted = ds
        .slice()
        .sort((a: any, b: any) => {
          const ta =
            (a?.pairedAt?.toDate?.() ? a.pairedAt.toDate().getTime() : null) ??
            (a?.updatedAt?.toDate?.() ? a.updatedAt.toDate().getTime() : null) ??
            (a?.createdAt?.toDate?.() ? a.createdAt.toDate().getTime() : 0);
          const tb =
            (b?.pairedAt?.toDate?.() ? b.pairedAt.toDate().getTime() : null) ??
            (b?.updatedAt?.toDate?.() ? b.updatedAt.toDate().getTime() : null) ??
            (b?.createdAt?.toDate?.() ? b.createdAt.toDate().getTime() : 0);
          return tb - ta;
        })
        .map((d: any) => d as Device);

      setDevices(sorted);
      setDevicesLoaded(true);

      if (sorted.length === 0) {
        setSelectedDeviceId('');
        userOverrodeDeviceSelectionRef.current = false;
        return;
      }

      const newestId = sorted[0].id;
      const selectedStillExists = !!selectedDeviceId && sorted.some((d) => d.id === selectedDeviceId);

      // Auto-select the newest device if:
      // - nothing selected yet, or selected device disappeared, or user never manually chose a device.
      if (!selectedDeviceId || !selectedStillExists || !userOverrodeDeviceSelectionRef.current) {
        if (newestId && newestId !== selectedDeviceId) {
          setSelectedDeviceId(newestId);
        }
      }
    });

    return () => unsubscribe();
  }, [user, selectedDeviceId]);

  // Prompt pairing via modal if user has no paired devices
  useEffect(() => {
    if (!user) {
      hasAutoOpenedPairModal.current = false;
      setShowPairModal(false);
      return;
    }
    if (!devicesLoaded) return;
    if (hasAutoOpenedPairModal.current) return;
    if (devices.length === 0) {
      setShowPairModal(true);
      hasAutoOpenedPairModal.current = true;
    }
  }, [user, devicesLoaded, devices.length]);

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
      setActiveFocusSession({ id: doc0.id, ...doc0.data() } as FocusSession);
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
      setToastMessage("Chat deleted");
    } catch (error) {
      console.error('Error deleting chat:', error);
      setToastMessage("Error deleting chat");
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
      await signOut(auth);
      setCourses([]);
      setSessions([]);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
      setDevices([]);
      setSelectedDeviceId('');
      userOverrodeDeviceSelectionRef.current = false;
      setActiveFocusSession(null);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  const handleClaimDeviceFromModal = async () => {
    if (!user || !claimCode.trim()) return;
    const submittedCode = claimCode.trim();
    setFocusBusy(true);
    try {
      await claimDevice(submittedCode, user.uid);
      setToastMessage("Device paired. Turn on your Pi agent to complete pairing.");
      setClaimCode('');
      setShowPairModal(false);
    } catch (error) {
      console.error('Error claiming device:', error);
      setToastMessage("Error pairing device");
    } finally {
      setFocusBusy(false);
    }
  };

  const openPairModalFromSettings = () => {
    setSettingsOpen(false);
    setShowPairModal(true);
  };

  const handleStartFocus = async () => {
    if (!user) return;
    if (!selectedDeviceId) {
      setToastMessage("No paired device. Pair a device first.");
      setShowPairModal(true);
      return;
    }
    // Optional: link focus session to the currently expanded course/session (chapter)
    const courseId = expandedCourseId || undefined;
    const sessionId = expandedSessionId || undefined;

    // Calibration gate: do not start focus until calibration completes
    setPendingFocusStart({ deviceId: selectedDeviceId, courseId, sessionId });
    setShowCalibrationModal(true);
  };

  const startFocusAfterCalibration = async () => {
    if (!user) return;
    if (!pendingFocusStart) return;

    setFocusBusy(true);
    try {
      const res = await startFocusSession({
        userId: user.uid,
        deviceId: pendingFocusStart.deviceId,
        courseId: pendingFocusStart.courseId,
        sessionId: pendingFocusStart.sessionId,
      });
      setToastMessage(`Focus tracking started (${res.focusSessionId.slice(0, 6)}...)`);
    } catch (error) {
      console.error('Error starting focus session:', error);
      setToastMessage("Error starting focus tracking");
    } finally {
      setFocusBusy(false);
      setPendingFocusStart(null);
    }
  };

  const handleStopFocus = async () => {
    if (!user || !activeFocusSession) return;
    setFocusBusy(true);
    try {
      await stopFocusSession({ userId: user.uid, focusSessionId: activeFocusSession.id, deviceId: activeFocusSession.deviceId });
      setToastMessage("Focus tracking stopped. Waiting for Pi summary...");
    } catch (error) {
      console.error('Error stopping focus session:', error);
      setToastMessage("Error stopping focus tracking");
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
                  ? 'Visualize your focus sessions uploaded from the Pi.'
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
                          className="chat-settings-item"
                          role="menuitem"
                          onClick={openPairModalFromSettings}
                          disabled={focusBusy}
                        >
                          Pair device
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
                    <span>Device {activeFocusSession.deviceId.slice(0, 8)}…</span>
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
            {!cameraPreviewAfterCalibration ? (
              <div className="preview-sidebar-empty">
                <p>
                  Start Focus to run the usual calibration. Once calibration is complete, the live preview will stay here while you study.
                </p>
              </div>
            ) : (
              <PiCalibrationPreview variant="embedded" mode="monitor" autoStart />
            )}
          </div>
        </div>
      )}
      
      {toastMessage && (
        <div className="toast-notification">
          {toastMessage}
        </div>
      )}

      {showPairModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Pair a device"
          onMouseDown={(e) => {
            // Close when clicking the backdrop (but not when interacting with the modal)
            if (e.target === e.currentTarget) setShowPairModal(false);
          }}
        >
          <div className="modal">
            <div className="modal-header">
              <div>
                <h3 className="modal-title">Pair your Raspberry Pi</h3>
                <p className="modal-subtitle">Enter the claim code shown by the Pi agent.</p>
              </div>
              <button
                className="modal-close"
                onClick={() => setShowPairModal(false)}
                aria-label="Close"
                type="button"
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-row">
                <input
                  value={claimCode}
                  onChange={(e) => setClaimCode(e.target.value)}
                  placeholder="e.g. ABCD-1234"
                  className="modal-field"
                  disabled={focusBusy}
                  autoFocus
                />
                <button
                  onClick={handleClaimDeviceFromModal}
                  className="modal-btn modal-btn-primary"
                  disabled={focusBusy || !claimCode.trim()}
                  type="button"
                >
                  Pair
                </button>
              </div>

              <div className="modal-hint">
                After pairing, keep the Pi agent running so it can complete pairing and start receiving focus sessions.
              </div>
            </div>
          </div>
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
                <p className="modal-subtitle">Align your camera before tracking. This preview runs on your local network.</p>
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
              <PiCalibrationPreview
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
