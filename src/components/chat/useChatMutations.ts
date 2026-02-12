import type { Dispatch, FormEvent, SetStateAction } from 'react';
import type { User } from 'firebase/auth';
import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  serverTimestamp,
  setDoc,
  updateDoc,
} from 'firebase/firestore';
import { db } from '../../firebase-config';
import { getAIResponse } from '../../services/genkit-service';
import type { Message } from './useChatCollections';

type ToastVariant = 'success' | 'warning' | 'info';

type UseChatMutationsParams = {
  user: User | null;
  expandedCourseId: string | null;
  expandedSessionId: string | null;
  selectedChatId: string | null;
  input: string;
  newCourseName: string;
  newSessionName: string;
  setInput: (value: string) => void;
  setLoading: (value: boolean) => void;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  setNewCourseName: (value: string) => void;
  setIsCreatingCourse: (value: boolean) => void;
  setNewSessionName: (value: string) => void;
  setIsCreatingSession: (value: boolean) => void;
  setExpandedSessionId: (value: string | null) => void;
  setSelectedChatId: (value: string | null) => void;
  setEditingChatId: (value: string | null) => void;
  enableAutoScroll: () => void;
  showToast: (message: string, variant?: ToastVariant) => void;
};

export function useChatMutations({
  user,
  expandedCourseId,
  expandedSessionId,
  selectedChatId,
  input,
  newCourseName,
  newSessionName,
  setInput,
  setLoading,
  setMessages,
  setNewCourseName,
  setIsCreatingCourse,
  setNewSessionName,
  setIsCreatingSession,
  setExpandedSessionId,
  setSelectedChatId,
  setEditingChatId,
  enableAutoScroll,
  showToast,
}: UseChatMutationsParams) {
  const handleCreateCourse = async (e: FormEvent) => {
    e.preventDefault();
    if (!user || !newCourseName.trim()) return;
    try {
      await addDoc(collection(db, 'courses'), {
        userId: user.uid,
        name: newCourseName.trim(),
        createdAt: serverTimestamp(),
      });
      setNewCourseName('');
      setIsCreatingCourse(false);
    } catch (error) {
      console.error('Error creating course:', error);
    }
  };

  const handleCreateSession = async (e: FormEvent) => {
    e.preventDefault();
    if (!user || !expandedCourseId || !newSessionName.trim()) return;
    try {
      const docRef = await addDoc(collection(db, 'sessions'), {
        userId: user.uid,
        courseId: expandedCourseId,
        name: newSessionName.trim(),
        createdAt: serverTimestamp(),
        focusScore: 0,
      });
      setNewSessionName('');
      setIsCreatingSession(false);
      setExpandedSessionId(docRef.id);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const handleCreateChat = async () => {
    if (!user || !expandedCourseId || !expandedSessionId) return;
    try {
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
      console.error('Error creating chat:', error);
    }
  };

  const handleSend = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user || !selectedChatId) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);
    enableAutoScroll();

    try {
      await addDoc(collection(db, 'messages'), {
        text: userMessage,
        userId: user.uid,
        userName: user.email?.split('@')[0] || 'User',
        sessionId: selectedChatId,
        createdAt: serverTimestamp(),
        isAI: false,
      });

      await updateDoc(doc(db, 'chats', selectedChatId), {
        lastMessageAt: serverTimestamp(),
      });

      const tempId = `temp-${Date.now()}`;
      setMessages((prev) => [...prev, {
        id: tempId,
        text: '',
        userId: user.uid,
        userName: 'AI Study Buddy',
        sessionId: selectedChatId,
        createdAt: new Date(),
        isAI: true,
      }]);

      let streamingText = '';
      const response = await getAIResponse(userMessage, selectedChatId, user.uid, (chunk) => {
        streamingText += chunk;
        setMessages((prev) => prev.map((m) => (m.id === tempId ? { ...m, text: streamingText } : m)));
      });

      setMessages((prev) => prev.filter((m) => m.id !== tempId));
      await addDoc(collection(db, 'messages'), {
        text: response.text,
        userId: user.uid,
        userName: 'AI Study Buddy',
        sessionId: selectedChatId,
        createdAt: serverTimestamp(),
        isAI: true,
        model: response.model,
      });

      await updateDoc(doc(db, 'chats', selectedChatId), {
        lastMessageAt: serverTimestamp(),
      });
    } catch (error) {
      console.error('Error sending message:', error);
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
      showToast('Chat deleted', 'success');
    } catch (error) {
      console.error('Error deleting chat:', error);
      showToast('Error deleting chat', 'warning');
    }
  };

  return {
    handleCreateCourse,
    handleCreateSession,
    handleCreateChat,
    handleSend,
    handleUpdateChatName,
    handleDeleteChat,
  };
}
