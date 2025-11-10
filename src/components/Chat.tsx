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
import { getAIResponse } from '../services/openai';
import './Chat.css';

interface Message {
  id: string;
  text: string;
  userId: string;
  userName: string;
  sessionId: string;
  createdAt: Date | null;
  isAI?: boolean;
  model?: string;
}

interface ChatSession {
  id: string;
  name: string;
  userId: string;
  createdAt: Date | null;
  lastMessageAt: Date | null;
}

interface ChatProps {
  user: User | null;
}

export default function Chat({ user }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const hasCreatedDefaultRef = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!user) {
      setMessages([]);
      setSessions([]);
      setCurrentSessionId(null);
      hasCreatedDefaultRef.current = false;
      return;
    }

    const sessionsQuery = query(
      collection(db, 'sessions'),
      where('userId', '==', user.uid),
      orderBy('lastMessageAt', 'desc')
    );

    const unsubscribeSessions = onSnapshot(sessionsQuery, (querySnapshot) => {
      const newSessions: ChatSession[] = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        newSessions.push({
          id: doc.id,
          name: data.name || 'New Chat',
          userId: data.userId,
          createdAt: data.createdAt instanceof Timestamp 
            ? data.createdAt.toDate() 
            : null,
          lastMessageAt: data.lastMessageAt instanceof Timestamp 
            ? data.lastMessageAt.toDate() 
            : null
        });
      });
      setSessions(newSessions);

      if (newSessions.length > 0 && !currentSessionId) {
        setCurrentSessionId(newSessions[0].id);
        hasCreatedDefaultRef.current = false;
      } else if (newSessions.length === 0 && user && !currentSessionId && !hasCreatedDefaultRef.current) {
        hasCreatedDefaultRef.current = true;
        const createDefaultSession = async () => {
          try {
            const newSession = await addDoc(collection(db, 'sessions'), {
              name: 'Chat 1',
              userId: user.uid,
              createdAt: serverTimestamp(),
              lastMessageAt: serverTimestamp()
            });
            setCurrentSessionId(newSession.id);
          } catch (error) {
            console.error('Error creating default session:', error);
            hasCreatedDefaultRef.current = false;
          }
        };
        createDefaultSession();
      }
    });

    return () => unsubscribeSessions();
  }, [user]);

  useEffect(() => {
    if (!user || !currentSessionId) {
      setMessages([]);
      return;
    }

    const messagesQuery = query(
      collection(db, 'messages'),
      where('sessionId', '==', currentSessionId),
      where('userId', '==', user.uid),
      orderBy('createdAt', 'asc')
    );

    const unsubscribe = onSnapshot(messagesQuery, (querySnapshot) => {
      const newMessages: Message[] = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        
        let messageText = data.text;
        if (typeof messageText === 'object' && messageText !== null) {
          messageText = messageText.text || messageText.content || JSON.stringify(messageText);
        }
        messageText = String(messageText || '');
        
        newMessages.push({
          id: doc.id,
          text: messageText,
          userId: data.userId,
          userName: data.userName || 'User',
          sessionId: data.sessionId,
          createdAt: data.createdAt instanceof Timestamp 
            ? data.createdAt.toDate() 
            : null,
          isAI: data.isAI || false,
          model: data.model || undefined
        });
      });
      setMessages(newMessages);
    });

    return () => unsubscribe();
  }, [user, currentSessionId]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || !user || !currentSessionId) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    try {
      await addDoc(collection(db, 'messages'), {
        text: userMessage,
        userId: user.uid,
        userName: user.email?.split('@')[0] || 'User',
        sessionId: currentSessionId,
        createdAt: serverTimestamp(),
        isAI: false
      });

      await updateSessionLastMessage(currentSessionId);

      const recentMessages = messages.slice(-10);
      const conversationHistory = recentMessages.map(msg => ({
        role: (!msg.isAI ? 'user' : 'assistant') as 'user' | 'assistant',
        content: msg.text
      }));

      try {
        const aiResponse = await getAIResponse(userMessage, conversationHistory);
        
        await addDoc(collection(db, 'messages'), {
          text: aiResponse.text,
          userId: user.uid,
          userName: 'AI Study Buddy',
          sessionId: currentSessionId,
          createdAt: serverTimestamp(),
          isAI: true,
          model: aiResponse.model
        });

        await updateSessionLastMessage(currentSessionId);
      } catch (aiError) {
        const errorMessage = aiError instanceof Error 
          ? `Error: ${aiError.message}` 
          : 'Sorry, I encountered an error. Please check your OpenAI API key configuration.';
        
        await addDoc(collection(db, 'messages'), {
          text: errorMessage,
          userId: user.uid,
          userName: 'AI Study Buddy',
          sessionId: currentSessionId,
          createdAt: serverTimestamp(),
          isAI: true,
          model: 'error'
        });
      }

      setLoading(false);
    } catch (error) {
      console.error('Error sending message:', error);
      setLoading(false);
    }
  };

  const updateSessionLastMessage = async (sessionId: string) => {
    try {
      const sessionRef = doc(db, 'sessions', sessionId);
      await updateDoc(sessionRef, {
        lastMessageAt: serverTimestamp()
      });
    } catch (error) {
      console.error('Error updating session:', error);
    }
  };

  const handleCreateSession = async (name?: string) => {
    if (!user) return;

    const sessionName = name || `Chat ${sessions.length + 1}`;
    
    try {
      const newSession = await addDoc(collection(db, 'sessions'), {
        name: sessionName,
        userId: user.uid,
        createdAt: serverTimestamp(),
        lastMessageAt: serverTimestamp()
      });

      setCurrentSessionId(newSession.id);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (!user || sessions.length <= 1) return;

    try {
      await deleteDoc(doc(db, 'sessions', sessionId));
      
      if (currentSessionId === sessionId) {
        const remainingSessions = sessions.filter(s => s.id !== sessionId);
        if (remainingSessions.length > 0) {
          setCurrentSessionId(remainingSessions[0].id);
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error);
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
    } catch (err: unknown) {
      if (err instanceof Error) {
        setAuthError(err.message);
      }
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
    } catch (err) {
      console.error('Error signing out:', err);
    }
  };

  if (!user) {
    return (
      <div className="chat-container">
        <div className="chat-header">
          <h2>AI Study Buddy Chat</h2>
          <p>Sign in to start chatting!</p>
        </div>
        {!showAuth ? (
          <div className="chat-placeholder">
            <div className="auth-prompt">
              <p>🚀 Welcome to AI Study Buddy!</p>
              <p style={{ fontSize: '0.9rem', opacity: 0.8, marginTop: '10px' }}>
                Sign in to start chatting with your AI study companion
              </p>
              <button 
                className="auth-toggle-button"
                onClick={() => setShowAuth(true)}
              >
                Sign In / Sign Up
              </button>
            </div>
          </div>
        ) : (
          <div className="chat-auth-form">
            <form onSubmit={handleAuth}>
              <h3>{isSignUp ? 'Create Account' : 'Sign In'}</h3>
              {authError && <div className="auth-error">{authError}</div>}
              <input
                type="email"
                placeholder="Email"
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                required
                className="auth-input"
              />
              <input
                type="password"
                placeholder="Password (min 6 characters)"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                required
                minLength={6}
                className="auth-input"
              />
              <button 
                type="submit" 
                className="auth-submit-button"
                disabled={authLoading}
              >
                {authLoading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}
              </button>
              <button
                type="button"
                className="auth-switch-button"
                onClick={() => {
                  setIsSignUp(!isSignUp);
                  setAuthError(null);
                }}
                disabled={authLoading}
              >
                {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
              </button>
              <button
                type="button"
                className="auth-cancel-button"
                onClick={() => {
                  setShowAuth(false);
                  setAuthError(null);
                }}
                disabled={authLoading}
              >
                Cancel
              </button>
            </form>
          </div>
        )}
      </div>
    );
  }

  const currentSession = sessions.find(s => s.id === currentSessionId);

  return (
    <div className="chat-container">
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <button 
            onClick={() => handleCreateSession()}
            className="new-chat-button"
            title="New Chat"
          >
            + New Chat
          </button>
        </div>
        <div className="sessions-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${currentSessionId === session.id ? 'active' : ''}`}
              onClick={() => setCurrentSessionId(session.id)}
            >
              <span className="session-name">{session.name}</span>
              {sessions.length > 1 && (
                <button
                  className="delete-session-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(session.id);
                  }}
                  title="Delete Chat"
                >
                  ×
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="chat-main">
        <div className="chat-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2>{currentSession?.name || 'AI Study Buddy Chat'}</h2>
              <p>Ask me anything!</p>
            </div>
            <button 
              onClick={handleSignOut}
              className="sign-out-button"
              title="Sign Out"
            >
              Sign Out
            </button>
          </div>
        </div>
        
        <div className="chat-messages" ref={messagesContainerRef}>
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <p>Start a conversation!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message ${!message.isAI ? 'message-user' : 'message-ai'}`}
            >
              <div className="message-content">
              <div className="message-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="message-name">
                    {!message.isAI ? 'You' : (message.userName || 'AI Study Buddy')}
                  </span>
                  {message.model && message.isAI && (
                    <span className="message-model" title={`Model: ${message.model}`}>
                      {message.model}
                    </span>
                  )}
                </div>
                {message.createdAt && (
                  <span className="message-time">
                    {message.createdAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                )}
              </div>
              <div className="message-text">{message.text}</div>
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="message message-ai">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

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
          <button 
            type="submit" 
            className="chat-send-button"
            disabled={!input.trim() || loading}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </form>
      </div>
    </div>
  );
}

