/**
 * Hook for managing exam sessions
 */

import { useState, useRef, useCallback } from 'react';
import { ExamConfiguration, ExamSession, AnsweredQuestion } from '../../types';
import { getAdaptiveDifficulty } from '../../services/exam-service';

interface UseExamSessionParams {
  examConfig: ExamConfiguration;
  userId: string;
}

export function useExamSession({ examConfig, userId }: UseExamSessionParams) {
  const [examSession, setExamSession] = useState<ExamSession | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [currentQuestionStartTime, setCurrentQuestionStartTime] = useState<number | null>(null);
  const [timeRemainingMs, setTimeRemainingMs] = useState(examConfig.totalTimeMs);
  const timerIntervalRef = useRef<number | null>(null);

  // Initialize exam session
  const startExam = useCallback(() => {
    const sessionId = `exam-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const firstSection = examConfig.sections[0];

    const newSession: ExamSession = {
      id: sessionId,
      userId,
      examId: examConfig.id,
      examTitle: examConfig.title,
      startedAt: new Date(),
      answeredQuestions: [],
      currentSectionId: firstSection.id,
      currentQuestionIndex: 0,
      totalTimeUsedMs: 0,
      status: 'in-progress',
    };

    setExamSession(newSession);
    setIsActive(true);
    setTimeRemainingMs(examConfig.totalTimeMs);
    setCurrentQuestionStartTime(Date.now());

    // Start timer
    const startTime = Date.now();
    timerIntervalRef.current = window.setInterval(() => {
      setTimeRemainingMs(() => {
        const newTime = examConfig.totalTimeMs - (Date.now() - startTime);
        return Math.max(0, newTime);
      });
    }, 100);

    return newSession;
  }, [examConfig, userId]);

  // Submit answer
  const submitAnswer = useCallback(
    (
      questionId: string,
      selectedAnswerIndex: number,
      topic: string,
      correctAnswerIndex: number,
      confidence: 'low' | 'medium' | 'high',
      questionDifficulty: 'easy' | 'medium' | 'hard',
      evaluatedCorrect?: boolean,
    ) => {
      if (!examSession) return;

      const isCorrect = typeof evaluatedCorrect === 'boolean'
        ? evaluatedCorrect
        : selectedAnswerIndex === correctAnswerIndex;
      const timeSpentMs = currentQuestionStartTime ? Date.now() - currentQuestionStartTime : 0;

      const newAnswer: AnsweredQuestion = {
        questionId,
        topic,
        selectedAnswerIndex,
        isCorrect,
        difficulty: questionDifficulty,
        timeSpentMs,
        confidence,
      };

      setExamSession((prev: ExamSession | null) => {
        if (!prev) return prev;
        return {
          ...prev,
          answeredQuestions: [...prev.answeredQuestions, newAnswer],
          currentQuestionIndex: prev.currentQuestionIndex + 1,
          totalTimeUsedMs: prev.totalTimeUsedMs + timeSpentMs,
        };
      });

      setCurrentQuestionStartTime(Date.now());
    },
    [examSession, currentQuestionStartTime]
  );

  // Complete exam
  const completeExam = useCallback(() => {
    if (!examSession) return;

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    setExamSession((prev: ExamSession | null) => {
      if (!prev) return prev;
      return {
        ...prev,
        status: 'completed',
        completedAt: new Date(),
      };
    });

    setIsActive(false);
    return { ...examSession, status: 'completed' as const, completedAt: new Date() };
  }, [examSession]);

  // Abandon exam
  const abandonExam = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    setExamSession((prev: ExamSession | null) => {
      if (!prev) return prev;
      return {
        ...prev,
        status: 'abandoned',
      };
    });

    setIsActive(false);
  }, []);

  // Get next difficulty
  const getNextDifficulty = useCallback(() => {
    if (!examSession) return 'medium';
    return getAdaptiveDifficulty(examSession, examConfig.adaptiveDifficulty);
  }, [examSession, examConfig.adaptiveDifficulty]);

  // Get current section
  const getCurrentSection = useCallback(() => {
    if (!examSession) return null;
    return examConfig.sections.find((s: any) => s.id === examSession.currentSectionId);
  }, [examSession, examConfig.sections]);

  // Get current question
  const getCurrentQuestion = useCallback(() => {
    if (!examSession) return null;
    const questionPool = examConfig.sections.flatMap((section) => section.questions);
    const questionLimit = examConfig.sections.reduce(
      (sum, section) => sum + Math.min(section.totalQuestionsTarget, section.questions.length),
      0,
    );
    const answeredIds = new Set(examSession.answeredQuestions.map((a) => a.questionId));

    if (answeredIds.size >= questionLimit) {
      return null;
    }

    const unanswered = questionPool.filter((q) => !answeredIds.has(q.id));
    if (unanswered.length === 0) {
      return null;
    }

    const difficulty = getNextDifficulty();
    const preferred = unanswered.filter((q) => q.initialDifficulty === difficulty);
    return preferred[0] ?? unanswered[0] ?? null;
  }, [examConfig.sections, examSession, getNextDifficulty]);

  return {
    examSession,
    isActive,
    timeRemainingMs,
    startExam,
    submitAnswer,
    completeExam,
    abandonExam,
    getNextDifficulty,
    getCurrentSection,
    getCurrentQuestion,
  };
}
