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
  ExamCompletionReason,
  ExamConfidence,
  ExamSimulation,
  ExamSimulationDifficultyBreakdown,
  ExamSimulationFocusInsight,
  ExamSimulationQuestion,
  ExamSimulationRecap,
  ExamSimulationResponse,
  ExamSimulationSection,
  ExamSimulationSectionResult,
  ExamSimulationStatus,
  ExamSimulationWeakTopic,
  StudyDifficulty,
} from '../../types';
import {
  createExamSimulation,
  finishExamSimulation,
  startExamSimulation,
  submitExamSimulationAnswer,
} from '../../services/exam-simulation-service';

type ToastVariant = 'success' | 'warning' | 'info';

type CurrentChatLike = {
  id: string;
  courseId: string;
  sessionId: string;
};

type UseExamSimulationParams = {
  user: User | null;
  selectedChatId: string | null;
  currentChat: CurrentChatLike | null;
  selectedModel: string;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function normalizeStatus(value: unknown): ExamSimulationStatus {
  if (
    value === 'generating' ||
    value === 'ready' ||
    value === 'in_progress' ||
    value === 'completed' ||
    value === 'timed_out' ||
    value === 'abandoned' ||
    value === 'failed'
  ) {
    return value;
  }
  return 'ready';
}

function normalizeDifficulty(value: unknown): StudyDifficulty {
  if (value === 'easy' || value === 'medium' || value === 'hard') return value;
  return 'medium';
}

function normalizeConfidence(value: unknown): ExamConfidence {
  if (value === 'low' || value === 'medium' || value === 'high') return value;
  return 'medium';
}

function toDate(value: unknown): Date | null {
  if (value instanceof Timestamp) return value.toDate();
  if (value instanceof Date) return value;
  return null;
}

function parseSections(value: unknown): ExamSimulationSection[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (
        typeof item.id !== 'string' ||
        typeof item.label !== 'string' ||
        typeof item.questionTargetCount !== 'number' ||
        typeof item.targetDurationSec !== 'number' ||
        typeof item.cumulativeTargetSec !== 'number'
      ) {
        return null;
      }
      return {
        id: item.id,
        label: item.label,
        questionTargetCount: item.questionTargetCount,
        targetDurationSec: item.targetDurationSec,
        cumulativeTargetSec: item.cumulativeTargetSec,
      } satisfies ExamSimulationSection;
    })
    .filter((section): section is ExamSimulationSection => section !== null);
}

function parseQuestions(value: unknown): ExamSimulationQuestion[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (
        typeof item.id !== 'string' ||
        typeof item.prompt !== 'string' ||
        typeof item.sectionId !== 'string'
      ) {
        return null;
      }
      const options = Array.isArray(item.options)
        ? item.options.filter((option): option is string => typeof option === 'string')
        : [];
      if (options.length === 0) return null;
      const sourceIds = Array.isArray(item.sourceIds)
        ? item.sourceIds.filter((sourceId): sourceId is string => typeof sourceId === 'string')
        : [];
      return {
        id: item.id,
        prompt: item.prompt,
        options,
        difficulty: normalizeDifficulty(item.difficulty),
        topic: typeof item.topic === 'string' ? item.topic : 'General review',
        sourceIds,
        sectionId: item.sectionId,
        orderIndex: typeof item.orderIndex === 'number' ? item.orderIndex : 0,
      } satisfies ExamSimulationQuestion;
    })
    .filter((question): question is ExamSimulationQuestion => question !== null)
    .sort((a, b) => a.orderIndex - b.orderIndex);
}

function parseResponses(value: unknown): ExamSimulationResponse[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (typeof item.questionId !== 'string' || typeof item.sectionId !== 'string') return null;
      return {
        questionId: item.questionId,
        questionOrder: typeof item.questionOrder === 'number' ? item.questionOrder : 0,
        selectedOptionIndex: typeof item.selectedOptionIndex === 'number' ? item.selectedOptionIndex : 0,
        confidence: normalizeConfidence(item.confidence),
        elapsedSec: typeof item.elapsedSec === 'number' ? item.elapsedSec : 0,
        answeredAt: toDate(item.answeredAt),
        difficulty: normalizeDifficulty(item.difficulty),
        topic: typeof item.topic === 'string' ? item.topic : 'General review',
        sectionId: item.sectionId,
      } satisfies ExamSimulationResponse;
    })
    .filter((response): response is ExamSimulationResponse => response !== null)
    .sort((a, b) => a.questionOrder - b.questionOrder);
}

function parseWeakTopics(value: unknown): ExamSimulationWeakTopic[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (typeof item.topic !== 'string') return null;
      return {
        topic: item.topic,
        accuracyPercent: typeof item.accuracyPercent === 'number' ? item.accuracyPercent : 0,
        questionCount: typeof item.questionCount === 'number' ? item.questionCount : 0,
        correctCount: typeof item.correctCount === 'number' ? item.correctCount : 0,
      } satisfies ExamSimulationWeakTopic;
    })
    .filter((topic): topic is ExamSimulationWeakTopic => topic !== null);
}

function parseDifficultyBreakdown(value: unknown): ExamSimulationDifficultyBreakdown[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      return {
        difficulty: normalizeDifficulty(item.difficulty),
        accuracyPercent: typeof item.accuracyPercent === 'number' ? item.accuracyPercent : 0,
        questionCount: typeof item.questionCount === 'number' ? item.questionCount : 0,
        correctCount: typeof item.correctCount === 'number' ? item.correctCount : 0,
      } satisfies ExamSimulationDifficultyBreakdown;
    })
    .filter((breakdown): breakdown is ExamSimulationDifficultyBreakdown => breakdown !== null);
}

function parseSectionResults(value: unknown): ExamSimulationSectionResult[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== 'object') return null;
      const item = row as Record<string, unknown>;
      if (typeof item.sectionId !== 'string' || typeof item.label !== 'string') return null;
      return {
        sectionId: item.sectionId,
        label: item.label,
        targetDurationSec: typeof item.targetDurationSec === 'number' ? item.targetDurationSec : 0,
        actualElapsedSec: typeof item.actualElapsedSec === 'number' ? item.actualElapsedSec : 0,
        overrunSec: typeof item.overrunSec === 'number' ? item.overrunSec : 0,
        answeredCount: typeof item.answeredCount === 'number' ? item.answeredCount : 0,
      } satisfies ExamSimulationSectionResult;
    })
    .filter((result): result is ExamSimulationSectionResult => result !== null);
}

function parseFocusInsight(value: unknown): ExamSimulationFocusInsight | null {
  if (!value || typeof value !== 'object') return null;
  const item = value as Record<string, unknown>;
  if (typeof item.focusSessionId !== 'string') return null;
  return {
    focusSessionId: item.focusSessionId,
    focusPercent: typeof item.focusPercent === 'number' ? item.focusPercent : 0,
    distractions: typeof item.distractions === 'number' ? item.distractions : 0,
    firstDriftOffsetSec: typeof item.firstDriftOffsetSec === 'number' ? item.firstDriftOffsetSec : null,
  };
}

function parseRecap(value: unknown): ExamSimulationRecap | null {
  if (!value || typeof value !== 'object') return null;
  const item = value as Record<string, unknown>;
  return {
    answeredCount: typeof item.answeredCount === 'number' ? item.answeredCount : 0,
    totalQuestionCount: typeof item.totalQuestionCount === 'number' ? item.totalQuestionCount : 0,
    correctCount: typeof item.correctCount === 'number' ? item.correctCount : 0,
    scorePercent: typeof item.scorePercent === 'number' ? item.scorePercent : 0,
    completionPercent: typeof item.completionPercent === 'number' ? item.completionPercent : 0,
    overconfidenceMisses: typeof item.overconfidenceMisses === 'number' ? item.overconfidenceMisses : 0,
    weakTopics: parseWeakTopics(item.weakTopics),
    accuracyByDifficulty: parseDifficultyBreakdown(item.accuracyByDifficulty),
    sectionResults: parseSectionResults(item.sectionResults),
    timeLossMoments: Array.isArray(item.timeLossMoments)
      ? item.timeLossMoments.filter((entry): entry is string => typeof entry === 'string')
      : [],
    recoveryPlan: Array.isArray(item.recoveryPlan)
      ? item.recoveryPlan.filter((entry): entry is string => typeof entry === 'string')
      : [],
    weakTopicSummary: typeof item.weakTopicSummary === 'string' ? item.weakTopicSummary : '',
    focusInsight: parseFocusInsight(item.focusInsight),
  };
}

export function useExamSimulation({
  user,
  selectedChatId,
  currentChat,
  selectedModel,
  showToast,
}: UseExamSimulationParams) {
  const [examSimulations, setExamSimulations] = useState<ExamSimulation[]>([]);
  const [activeExamId, setActiveExamId] = useState<string | null>(null);
  const [examGenerating, setExamGenerating] = useState(false);
  const [examActionBusy, setExamActionBusy] = useState(false);

  useEffect(() => {
    if (!user || !selectedChatId) {
      setExamSimulations([]);
      setActiveExamId(null);
      return;
    }

    const q = query(
      collection(db, 'examSimulations'),
      where('userId', '==', user.uid),
      where('chatId', '==', selectedChatId),
      orderBy('createdAt', 'desc'),
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const next: ExamSimulation[] = snapshot.docs.map((docSnap) => {
        const data = docSnap.data();
        return {
          id: docSnap.id,
          userId: typeof data.userId === 'string' ? data.userId : '',
          chatId: typeof data.chatId === 'string' ? data.chatId : '',
          courseId: typeof data.courseId === 'string' ? data.courseId : null,
          sessionId: typeof data.sessionId === 'string' ? data.sessionId : null,
          status: normalizeStatus(data.status),
          preset: 'standard_mock',
          durationSec: typeof data.durationSec === 'number' ? data.durationSec : 0,
          servedQuestionCount: typeof data.servedQuestionCount === 'number' ? data.servedQuestionCount : 0,
          questionBankCount: typeof data.questionBankCount === 'number' ? data.questionBankCount : 0,
          model: typeof data.model === 'string' ? data.model : 'unknown',
          sections: parseSections(data.sections),
          servedQuestions: parseQuestions(data.servedQuestions),
          responses: parseResponses(data.responses),
          currentQuestionId: typeof data.currentQuestionId === 'string' ? data.currentQuestionId : null,
          recap: parseRecap(data.recap),
          errorMessage: typeof data.errorMessage === 'string' ? data.errorMessage : null,
          createdAt: toDate(data.createdAt),
          updatedAt: toDate(data.updatedAt),
          startedAt: toDate(data.startedAt),
          endsAt: toDate(data.endsAt),
          finishedAt: toDate(data.finishedAt),
        } satisfies ExamSimulation;
      });

      setExamSimulations(next);
      if (next.length === 0) {
        setActiveExamId(null);
        return;
      }

      const hasCurrent = next.some((exam) => exam.id === activeExamId);
      if (hasCurrent) return;

      const inProgress = next.find((exam) => exam.status === 'in_progress');
      setActiveExamId(inProgress?.id ?? next[0].id);
    });

    return () => unsubscribe();
  }, [activeExamId, selectedChatId, user]);

  const activeExam = useMemo(
    () => examSimulations.find((exam) => exam.id === activeExamId) ?? (examSimulations[0] ?? null),
    [activeExamId, examSimulations],
  );

  const handleCreateExam = useCallback(async () => {
    if (!user || !selectedChatId || !currentChat) {
      showToast('Select a chat before generating a mock exam.', 'warning');
      return false;
    }

    setExamGenerating(true);
    try {
      const result = await createExamSimulation({
        userId: user.uid,
        chatId: selectedChatId,
        model: selectedModel,
      });
      setActiveExamId(result.examSimulationId);
      showToast('Mock exam generated', 'success');
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to generate mock exam';
      showToast(message, 'warning');
      return false;
    } finally {
      setExamGenerating(false);
    }
  }, [currentChat, selectedChatId, selectedModel, showToast, user]);

  const handleStartExam = useCallback(async (examSimulationId: string) => {
    if (!user) return false;
    setExamActionBusy(true);
    try {
      await startExamSimulation({
        userId: user.uid,
        examSimulationId,
      });
      setActiveExamId(examSimulationId);
      showToast('Mock exam started', 'success');
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start mock exam';
      showToast(message, 'warning');
      return false;
    } finally {
      setExamActionBusy(false);
    }
  }, [showToast, user]);

  const handleSubmitAnswer = useCallback(async (params: {
    examSimulationId: string;
    questionId: string;
    selectedOptionIndex: number;
    confidence: ExamConfidence;
    elapsedSec: number;
  }) => {
    if (!user) return false;
    setExamActionBusy(true);
    try {
      await submitExamSimulationAnswer({
        userId: user.uid,
        ...params,
      });
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to submit exam answer';
      showToast(message, 'warning');
      return false;
    } finally {
      setExamActionBusy(false);
    }
  }, [showToast, user]);

  const handleFinishExam = useCallback(async (params: {
    examSimulationId: string;
    completionReason: ExamCompletionReason;
    focusSessionId?: string;
  }) => {
    if (!user) return false;
    setExamActionBusy(true);
    try {
      await finishExamSimulation({
        userId: user.uid,
        ...params,
      });
      if (params.completionReason === 'submitted') {
        showToast('Mock exam submitted', 'success');
      } else if (params.completionReason === 'time_up') {
        showToast('Time is up. Finalizing your mock exam.', 'info');
      } else {
        showToast('Mock exam exited early', 'warning');
      }
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to finalize mock exam';
      showToast(message, 'warning');
      return false;
    } finally {
      setExamActionBusy(false);
    }
  }, [showToast, user]);

  return {
    examSimulations,
    activeExam,
    activeExamId,
    setActiveExamId,
    examGenerating,
    examActionBusy,
    handleCreateExam,
    handleStartExam,
    handleSubmitAnswer,
    handleFinishExam,
  };
}
