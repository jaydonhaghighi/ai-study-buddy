import { useState, useEffect, useRef } from 'react';
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

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

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
    if (!confirm('Are you sure you want to delete this chat?')) return;
    try {
      await deleteDoc(doc(db, 'chats', chatId));
      if (selectedChatId === chatId) setSelectedChatId(null);
    } catch (error) {
      console.error('Error deleting chat:', error);
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
    <div className="chat-container">
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
                <span className="icon">{expandedCourseId === course.id ? '▼' : '▶'}</span>
                <span className="name">{course.name}</span>
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
                        <span className="icon">{expandedSessionId === session.id ? '📂' : '📁'}</span>
                        <span className="name">{session.name}</span>
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
              <p>{currentChat ? 'AI Study Buddy' : 'Select a course and session to start chatting'}</p>
            </div>
            <button onClick={handleSignOut} className="sign-out-button">Sign Out</button>
          </div>
        </div>

        <div className="chat-messages" ref={messagesContainerRef}>
          {messages.length === 0 && !selectedChatId ? (
             <div className="chat-welcome"><p>Select or create a chat to begin.</p></div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className={`message ${!message.isAI ? 'message-user' : 'message-ai'}`}>
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-name">{!message.isAI ? 'You' : (message.userName || 'AI Study Buddy')}</span>
                    {message.model && message.isAI && <span className="message-model">{message.model}</span>}
                  </div>
                  <div className="message-text"><div className="plain-text">{message.text}</div></div>
                </div>
              </div>
            ))
          )}
          {loading && <div className="message message-ai"><div className="message-content"><div className="typing-indicator"><span></span><span></span><span></span></div></div></div>}
          <div ref={messagesEndRef} />
        </div>

        <form className="chat-input-form" onSubmit={handleSend}>
          <div className="chat-input-wrapper">
            <input
              type="text"
              className="chat-input"
              placeholder={selectedChatId ? "Type your message..." : "Select a chat first"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading || !selectedChatId}
            />
            <button type="submit" className="chat-send-button" disabled={!input.trim() || loading || !selectedChatId}>
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
