import { useState, useEffect, useRef } from 'react';
import { db, auth, storage } from '../firebase-config';
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
import { ref, uploadBytes, getDownloadURL, listAll, deleteObject, getMetadata } from 'firebase/storage';
import { getAIResponse, createGenkitChat } from '../services/genkit-service';
import { claimDevice, startFocusSession, stopFocusSession } from '../services/focus-service';
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

interface UploadedFile {
  id: string;
  name: string;
  url: string;
  type: string;
  size: number;
  uploadedAt: Date;
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

  // Files Sidebar State
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Device + Focus Tracking State
  const [claimCode, setClaimCode] = useState('');
  const [devices, setDevices] = useState<Device[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [activeFocusSession, setActiveFocusSession] = useState<FocusSession | null>(null);
  const [focusBusy, setFocusBusy] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      setSelectedDeviceId('');
      return;
    }

    const q = query(
      collection(db, 'devices'),
      where('pairedUserId', '==', user.uid)
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const ds = snapshot.docs.map(d => ({ id: d.id, ...d.data() } as Device));
      setDevices(ds);
      if (!selectedDeviceId && ds.length > 0) {
        setSelectedDeviceId(ds[0].id);
      }
    });

    return () => unsubscribe();
  }, [user, selectedDeviceId]);

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
  }, [user, selectedChatId]);

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

  // File Upload Handlers
  const handleFileUpload = async (files: FileList | null) => {
    if (!user || !selectedChatId || !files || files.length === 0) return;

    setUploading(true);
    const uploadPromises: Promise<void>[] = [];

    Array.from(files).forEach((file) => {
      const uploadPromise = (async () => {
        try {
          const fileRef = ref(storage, `chats/${selectedChatId}/${user.uid}/${Date.now()}_${file.name}`);
          await uploadBytes(fileRef, file);
          const url = await getDownloadURL(fileRef);
          
          const newFile: UploadedFile = {
            id: fileRef.fullPath,
            name: file.name,
            url,
            type: file.type,
            size: file.size,
            uploadedAt: new Date(),
          };
          
          setUploadedFiles(prev => [...prev, newFile]);
          setToastMessage(`${file.name} uploaded successfully`);
        } catch (error) {
          console.error('Error uploading file:', error);
          setToastMessage(`Error uploading ${file.name}`);
        }
      })();
      uploadPromises.push(uploadPromise);
    });

    await Promise.all(uploadPromises);
    setUploading(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileUpload(e.dataTransfer.files);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileUpload(e.target.files);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDeleteFile = async (fileId: string, fileName: string) => {
    if (!user) return;
    try {
      const fileRef = ref(storage, fileId);
      await deleteObject(fileRef);
      setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
      setToastMessage(`${fileName} deleted`);
    } catch (error) {
      console.error('Error deleting file:', error);
      setToastMessage(`Error deleting ${fileName}`);
    }
  };

  // Load files for selected chat
  useEffect(() => {
    if (!user || !selectedChatId) {
      setUploadedFiles([]);
      return;
    }

    const loadFiles = async () => {
      try {
        const filesRef = ref(storage, `chats/${selectedChatId}/${user.uid}`);
        const filesList = await listAll(filesRef);
        
        const filePromises = filesList.items.map(async (itemRef) => {
          const url = await getDownloadURL(itemRef);
          const metadata = await getMetadata(itemRef);
          // Extract original filename (remove timestamp prefix)
          const fileName = itemRef.name.replace(/^\d+_/, '');
          return {
            id: itemRef.fullPath,
            name: fileName,
            url,
            type: metadata.contentType || 'application/octet-stream',
            size: metadata.size || 0,
            uploadedAt: metadata.timeCreated ? new Date(metadata.timeCreated) : new Date(),
          } as UploadedFile;
        });

        const files = await Promise.all(filePromises);
        setUploadedFiles(files);
      } catch (error) {
        console.error('Error loading files:', error);
      }
    };

    loadFiles();
  }, [user, selectedChatId]);

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
      await signOut(auth);
      setCourses([]);
      setSessions([]);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
      setDevices([]);
      setSelectedDeviceId('');
      setActiveFocusSession(null);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  const handleClaimDevice = async () => {
    if (!user || !claimCode.trim()) return;
    setFocusBusy(true);
    try {
      await claimDevice(claimCode.trim(), user.uid);
      setToastMessage("Device paired. Turn on your Pi agent to complete pairing.");
      setClaimCode('');
    } catch (error) {
      console.error('Error claiming device:', error);
      setToastMessage("Error pairing device");
    } finally {
      setFocusBusy(false);
    }
  };

  const handleStartFocus = async () => {
    if (!user) return;
    if (!selectedDeviceId) {
      setToastMessage("Select a device first");
      return;
    }
    // Optional: link focus session to the currently expanded course/session (chapter)
    const courseId = expandedCourseId || undefined;
    const sessionId = expandedSessionId || undefined;

    setFocusBusy(true);
    try {
      const res = await startFocusSession({ userId: user.uid, deviceId: selectedDeviceId, courseId, sessionId });
      setToastMessage(`Focus tracking started (${res.focusSessionId.slice(0, 6)}...)`);
    } catch (error) {
      console.error('Error starting focus session:', error);
      setToastMessage("Error starting focus tracking");
    } finally {
      setFocusBusy(false);
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
      <div className="chat-container">
        <div className="chat-header">
           <h2>AI Study Buddy</h2>
           <p>Sign in to start learning!</p>
        </div>
        {!showAuth ? (
          <div className="chat-placeholder">
            <div className="auth-prompt">
              <p>🚀 Organize your learning with Courses & Sessions</p>
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
    <div className="chat-container files-sidebar-open">
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2>{currentChat?.name || 'Select a Chat'}</h2>
              <p>{currentChat ? 'AI Study Buddy' : 'Choose an existing chat or create a new one to get started'}</p>
            </div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    value={claimCode}
                    onChange={(e) => setClaimCode(e.target.value)}
                    placeholder="Enter Pi claim code"
                    className="auth-input"
                    style={{ width: 180 }}
                    disabled={focusBusy}
                  />
                  <button onClick={handleClaimDevice} className="auth-submit-button" disabled={focusBusy || !claimCode.trim()}>
                    Pair
                  </button>
                </div>

                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                    className="auth-input"
                    style={{ width: 180 }}
                    disabled={focusBusy || devices.length === 0}
                  >
                    {devices.length === 0 ? (
                      <option value="">No paired devices</option>
                    ) : (
                      devices.map(d => <option key={d.id} value={d.id}>{d.id.slice(0, 8)}...</option>)
                    )}
                  </select>

                  {!activeFocusSession ? (
                    <button onClick={handleStartFocus} className="auth-submit-button" disabled={focusBusy || !selectedDeviceId}>
                      Start Focus
                    </button>
                  ) : (
                    <button onClick={handleStopFocus} className="auth-cancel-button" disabled={focusBusy}>
                      Stop Focus
                    </button>
                  )}
                </div>
                {activeFocusSession && (
                  <div style={{ fontSize: 12, opacity: 0.8 }}>
                    Focus active on device {activeFocusSession.deviceId.slice(0, 8)}...
                  </div>
                )}
              </div>
              <button onClick={handleSignOut} className="sign-out-button">Sign Out</button>
            </div>
          </div>
        </div>

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
                    <div className="message-text"><div className="plain-text">{message.text}</div></div>
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
      </div>

      {/* Files Sidebar */}
      {user && (
        <div className="files-sidebar">
          <div className="files-sidebar-header">
            <h3>Files</h3>
          </div>

              <div 
                className={`files-drop-zone ${isDragging ? 'dragging' : ''} ${uploading ? 'uploading' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={handleFileInputChange}
                  style={{ display: 'none' }}
                  accept="*/*"
                />
                <div className="drop-zone-content">
                  <div className="drop-zone-icon">📁</div>
                  <p className="drop-zone-text">
                    {uploading ? 'Uploading...' : isDragging ? 'Drop files here' : 'Drag & drop files here'}
                  </p>
                  <button 
                    className="drop-zone-button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    Or click to browse
                  </button>
                </div>
              </div>

              <div className="files-list">
                {uploadedFiles.length === 0 ? (
                  <div className="files-empty">
                    <p>No files uploaded yet</p>
                  </div>
                ) : (
                  uploadedFiles.map((file) => (
                    <div key={file.id} className="file-item">
                      <div className="file-info">
                        <span className="file-icon">
                          {file.type.startsWith('image/') ? '🖼️' : 
                           file.type.includes('pdf') ? '📄' :
                           file.type.includes('word') || file.name.endsWith('.docx') ? '📝' :
                           file.type.includes('powerpoint') || file.name.endsWith('.pptx') ? '📊' :
                           '📎'}
                        </span>
                        <div className="file-details">
                          <span className="file-name" title={file.name}>{file.name}</span>
                          <span className="file-size">
                            {(file.size / 1024).toFixed(1)} KB
                          </span>
                        </div>
                      </div>
                      <div className="file-actions">
                        <a 
                          href={file.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="file-action-btn"
                          title="Open"
                        >
                          ↗
                        </a>
                        <button
                          onClick={() => handleDeleteFile(file.id, file.name)}
                          className="file-action-btn"
                          title="Delete"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
        </div>
      )}
      
      {toastMessage && (
        <div className="toast-notification">
          {toastMessage}
        </div>
      )}
    </div>
  );
}
