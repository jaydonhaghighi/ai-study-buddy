import { useCallback, useEffect, useMemo, useState } from 'react';
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
import type {
  ExamQuestion,
  Flashcard,
  FlashcardReviewRating,
  QuizQuestion,
  StudyDifficulty,
  StudySet,
  StudySource,
  StudySetStatus,
} from '../../types';
import {
  generateStudySet,
  submitFlashcardReview,
  type StudySetPreset,
} from '../../services/study-set-service';

type ToastVariant = 'success' | 'warning' | 'info';

type CurrentChatLike = {
  id: string;
  courseId: string;
  sessionId: string;
};

type UseStudySetsParams = {
  user: User | null;
  selectedChatId: string | null;
  currentChat: CurrentChatLike | null;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function normalizeStatus(value: unknown): StudySetStatus {
  if (value === 'generating' || value === 'ready' || value === 'failed') return value;
  return 'ready';
}

function normalizeDifficulty(value: unknown): StudyDifficulty {
  if (value === 'easy' || value === 'medium' || value === 'hard') return value;
  return 'medium';
}

function toDate(value: unknown): Date | null {
  if (value instanceof Timestamp) return value.toDate();
  if (value instanceof Date) return value;
  return null;
}

function parseSources(value: unknown): StudySource[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const source = row as Record<string, unknown>;
      if (
        typeof source.id !== 'string' ||
        (source.type !== 'chat' && source.type !== 'material') ||
        typeof source.label !== 'string' ||
        typeof source.snippet !== 'string'
      ) {
        return null;
      }
      return {
        id: source.id,
        type: source.type,
        label: source.label,
        snippet: source.snippet,
      } as StudySource;
    })
    .filter((item): item is StudySource => item !== null);
}

function parseQuiz(value: unknown): QuizQuestion[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      const questionType = item.questionType === 'mcq' ? 'mcq' : item.questionType === 'short' ? 'short' : null;
      if (
        typeof item.id !== 'string' ||
        !questionType ||
        typeof item.prompt !== 'string' ||
        typeof item.correctAnswer !== 'string' ||
        typeof item.explanation !== 'string'
      ) {
        return null;
      }

      const options = Array.isArray(item.options)
        ? item.options.filter((opt): opt is string => typeof opt === 'string')
        : [];
      const sourceIds = Array.isArray(item.sourceIds)
        ? item.sourceIds.filter((id): id is string => typeof id === 'string')
        : [];

      return {
        id: item.id,
        questionType,
        prompt: item.prompt,
        options,
        correctAnswer: item.correctAnswer,
        correctOptionIndex: typeof item.correctOptionIndex === 'number' ? item.correctOptionIndex : null,
        explanation: item.explanation,
        difficulty: normalizeDifficulty(item.difficulty),
        sourceIds,
      } as QuizQuestion;
    })
    .filter((item): item is QuizQuestion => item !== null);
}

function parseFlashcards(value: unknown): Flashcard[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (
        typeof item.id !== 'string' ||
        typeof item.front !== 'string' ||
        typeof item.back !== 'string'
      ) {
        return null;
      }

      const tags = Array.isArray(item.tags)
        ? item.tags.filter((tag): tag is string => typeof tag === 'string')
        : [];
      const sourceIds = Array.isArray(item.sourceIds)
        ? item.sourceIds.filter((id): id is string => typeof id === 'string')
        : [];

      return {
        id: item.id,
        front: item.front,
        back: item.back,
        tags,
        difficulty: normalizeDifficulty(item.difficulty),
        sourceIds,
        nextReviewAt: toDate(item.nextReviewAt),
        intervalDays: typeof item.intervalDays === 'number' ? item.intervalDays : 1,
        easeFactor: typeof item.easeFactor === 'number' ? item.easeFactor : 2.5,
        repetitions: typeof item.repetitions === 'number' ? item.repetitions : 0,
        lastReviewedAt: toDate(item.lastReviewedAt),
      } as Flashcard;
    })
    .filter((item): item is Flashcard => item !== null);
}

function parseExamQuestions(value: unknown): ExamQuestion[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (
        typeof item.id !== 'string' ||
        typeof item.prompt !== 'string' ||
        typeof item.modelAnswer !== 'string'
      ) {
        return null;
      }
      const rubric = Array.isArray(item.rubric)
        ? item.rubric.filter((entry): entry is string => typeof entry === 'string')
        : [];
      const sourceIds = Array.isArray(item.sourceIds)
        ? item.sourceIds.filter((id): id is string => typeof id === 'string')
        : [];
      return {
        id: item.id,
        prompt: item.prompt,
        rubric,
        modelAnswer: item.modelAnswer,
        difficulty: normalizeDifficulty(item.difficulty),
        sourceIds,
      } as ExamQuestion;
    })
    .filter((item): item is ExamQuestion => item !== null);
}

export function useStudySets({
  user,
  selectedChatId,
  currentChat,
  showToast,
}: UseStudySetsParams) {
  const [studySets, setStudySets] = useState<StudySet[]>([]);
  const [activeStudySetId, setActiveStudySetId] = useState<string | null>(null);
  const [studySetGenerating, setStudySetGenerating] = useState(false);
  const [reviewBusyCardId, setReviewBusyCardId] = useState<string | null>(null);

  useEffect(() => {
    if (!user || !selectedChatId) {
      setStudySets([]);
      setActiveStudySetId(null);
      return;
    }

    const q = query(
      collection(db, 'studySets'),
      where('userId', '==', user.uid),
      where('chatId', '==', selectedChatId),
      orderBy('createdAt', 'desc'),
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const next: StudySet[] = snapshot.docs.map((docSnap) => {
        const data = docSnap.data();
        return {
          id: docSnap.id,
          userId: typeof data.userId === 'string' ? data.userId : '',
          chatId: typeof data.chatId === 'string' ? data.chatId : '',
          courseId: typeof data.courseId === 'string' ? data.courseId : null,
          sessionId: typeof data.sessionId === 'string' ? data.sessionId : null,
          status: normalizeStatus(data.status),
          quizQuestions: parseQuiz(data.quizQuestions),
          flashcards: parseFlashcards(data.flashcards),
          examQuestions: parseExamQuestions(data.examQuestions),
          sources: parseSources(data.sources),
          model: typeof data.model === 'string' ? data.model : 'unknown',
          generationMs: typeof data.generationMs === 'number' ? data.generationMs : null,
          errorMessage: typeof data.errorMessage === 'string' ? data.errorMessage : null,
          createdAt: toDate(data.createdAt),
          updatedAt: toDate(data.updatedAt),
        } as StudySet;
      });

      setStudySets(next);
      if (next.length === 0) {
        setActiveStudySetId(null);
        return;
      }
      const activeExists = next.some((set) => set.id === activeStudySetId);
      if (!activeExists) {
        setActiveStudySetId(next[0].id);
      }
    });

    return () => unsubscribe();
  }, [activeStudySetId, selectedChatId, user]);

  const activeStudySet = useMemo(
    () => studySets.find((set) => set.id === activeStudySetId) ?? (studySets[0] ?? null),
    [activeStudySetId, studySets],
  );

  const handleGenerateStudySet = useCallback(async (preset: StudySetPreset) => {
    if (!user || !selectedChatId || !currentChat) {
      showToast('Select a chat before generating a study set.', 'warning');
      return;
    }

    setStudySetGenerating(true);
    try {
      const result = await generateStudySet({
        userId: user.uid,
        chatId: selectedChatId,
        preset,
      });
      setActiveStudySetId(result.studySetId);
      showToast('Study set generated', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to generate study set';
      showToast(message, 'warning');
    } finally {
      setStudySetGenerating(false);
    }
  }, [currentChat, selectedChatId, showToast, user]);

  const handleReviewFlashcard = useCallback(async (
    studySetId: string,
    cardId: string,
    rating: FlashcardReviewRating,
  ) => {
    if (!user) return;
    const busyId = `${studySetId}:${cardId}`;
    setReviewBusyCardId(busyId);

    try {
      await submitFlashcardReview({
        userId: user.uid,
        studySetId,
        cardId,
        rating,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save flashcard review';
      showToast(message, 'warning');
    } finally {
      setReviewBusyCardId(null);
    }
  }, [showToast, user]);

  return {
    studySets,
    activeStudySet,
    activeStudySetId,
    setActiveStudySetId,
    studySetGenerating,
    reviewBusyCardId,
    handleGenerateStudySet,
    handleReviewFlashcard,
  };
}
