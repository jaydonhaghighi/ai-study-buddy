import { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';
import {
  collection,
  onSnapshot,
  orderBy,
  query,
  Timestamp,
  where,
} from 'firebase/firestore';
import { db } from '../../firebase-config';

export type Message = {
  id: string;
  text: string;
  userId: string;
  userName: string;
  sessionId: string;
  createdAt: Date | null;
  isAI?: boolean;
  model?: string;
};

export type Course = {
  id: string;
  name: string;
  userId: string;
};

export type Session = {
  id: string;
  courseId: string;
  name: string;
  userId: string;
  createdAt: Date | null;
};

export type ChatSession = {
  id: string;
  sessionId: string;
  courseId: string;
  name: string;
  userId: string;
  createdAt: Date | null;
  lastMessageAt: Date | null;
};

type UseChatCollectionsParams = {
  user: User | null;
  expandedCourseId: string | null;
  expandedSessionId: string | null;
  selectedChatId: string | null;
  mainView: 'chat' | 'dashboard';
};

export function useChatCollections({
  user,
  expandedCourseId,
  expandedSessionId,
  selectedChatId,
  mainView,
}: UseChatCollectionsParams) {
  const [courses, setCourses] = useState<Course[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);

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
      setCourses(snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() } as Course)));
    });

    return () => unsubscribe();
  }, [user]);

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
      setSessions(snapshot.docs.map((doc) => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null,
        } as Session;
      }));
    });

    return () => unsubscribe();
  }, [user, expandedCourseId]);

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
      setChats(snapshot.docs.map((doc) => {
        const data = doc.data();
        return {
          id: doc.id,
          ...data,
          createdAt: data.createdAt instanceof Timestamp ? data.createdAt.toDate() : null,
          lastMessageAt: data.lastMessageAt instanceof Timestamp ? data.lastMessageAt.toDate() : null,
        } as ChatSession;
      }));
    });

    return () => unsubscribe();
  }, [user, expandedSessionId]);

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
          model: data.model,
        });
      });
      setMessages(msgs);
    });

    return () => unsubscribe();
  }, [user, selectedChatId, mainView]);

  const clearAllData = () => {
    setCourses([]);
    setSessions([]);
    setChats([]);
    setMessages([]);
  };

  return {
    courses,
    sessions,
    chats,
    messages,
    setMessages,
    clearAllData,
  };
}
