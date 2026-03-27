import { fetchFunctionsEndpoint } from './functions-http';
import type {
  ExamCompletionReason,
  ExamConfidence,
  ExamSimulationQuestion,
  ExamSimulationRecap,
  ExamSimulationStatus,
  StudyDifficulty,
} from '../types';

type RawExamQuestion = {
  id?: unknown;
  prompt?: unknown;
  options?: unknown;
  difficulty?: unknown;
  topic?: unknown;
  sourceIds?: unknown;
  sectionId?: unknown;
  orderIndex?: unknown;
};

function normalizeDifficulty(value: unknown): StudyDifficulty {
  if (value === 'easy' || value === 'medium' || value === 'hard') return value;
  return 'medium';
}

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

function parseQuestion(value: unknown): ExamSimulationQuestion | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as RawExamQuestion;
  if (typeof row.id !== 'string' || typeof row.prompt !== 'string' || typeof row.sectionId !== 'string') {
    return null;
  }
  const options = Array.isArray(row.options)
    ? row.options.filter((option): option is string => typeof option === 'string')
    : [];
  if (options.length === 0) return null;
  const sourceIds = Array.isArray(row.sourceIds)
    ? row.sourceIds.filter((id): id is string => typeof id === 'string')
    : [];
  return {
    id: row.id,
    prompt: row.prompt,
    options,
    difficulty: normalizeDifficulty(row.difficulty),
    topic: typeof row.topic === 'string' ? row.topic : 'General review',
    sourceIds,
    sectionId: row.sectionId,
    orderIndex: typeof row.orderIndex === 'number' ? row.orderIndex : 0,
  };
}

export async function createExamSimulation(params: {
  userId: string;
  chatId: string;
  model?: string;
}): Promise<{ examSimulationId: string }> {
  const response = await fetchFunctionsEndpoint('/examSimulationCreate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const payload = (await response.json()) as { examSimulationId?: string };
  if (typeof payload.examSimulationId !== 'string' || payload.examSimulationId.length === 0) {
    throw new Error('Exam simulation ID missing in response');
  }
  return { examSimulationId: payload.examSimulationId };
}

export async function startExamSimulation(params: {
  userId: string;
  examSimulationId: string;
}): Promise<{
  status: ExamSimulationStatus;
  currentQuestion: ExamSimulationQuestion | null;
  startedAt: string | null;
  endsAt: string | null;
}> {
  const response = await fetchFunctionsEndpoint('/examSimulationStart', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const payload = await response.json();
  return {
    status: normalizeStatus(payload?.status),
    currentQuestion: parseQuestion(payload?.currentQuestion),
    startedAt: typeof payload?.startedAt === 'string' ? payload.startedAt : null,
    endsAt: typeof payload?.endsAt === 'string' ? payload.endsAt : null,
  };
}

export async function submitExamSimulationAnswer(params: {
  userId: string;
  examSimulationId: string;
  questionId: string;
  selectedOptionIndex: number;
  confidence: ExamConfidence;
  elapsedSec: number;
}): Promise<{
  status: ExamSimulationStatus;
  done: boolean;
  currentQuestion: ExamSimulationQuestion | null;
}> {
  const response = await fetchFunctionsEndpoint('/examSimulationAnswer', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const payload = await response.json();
  return {
    status: normalizeStatus(payload?.status),
    done: payload?.done === true,
    currentQuestion: parseQuestion(payload?.currentQuestion),
  };
}

export async function finishExamSimulation(params: {
  userId: string;
  examSimulationId: string;
  completionReason: ExamCompletionReason;
  focusSessionId?: string;
}): Promise<{
  status: ExamSimulationStatus;
  recap: ExamSimulationRecap | null;
}> {
  const response = await fetchFunctionsEndpoint('/examSimulationFinish', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const payload = await response.json();
  return {
    status: normalizeStatus(payload?.status),
    recap: (payload?.recap as ExamSimulationRecap | null) ?? null,
  };
}
