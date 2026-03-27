import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore, type Firestore } from "firebase-admin/firestore";
import { getStorage } from "firebase-admin/storage";
import crypto from "crypto";
import { genkit } from "genkit/beta";
import { openAI } from "@genkit-ai/compat-oai/openai";
import type {
  GenerateOptions,
  GenerateResponse,
  GenerateResponseChunk,
} from "genkit";
import { PDFParse } from "pdf-parse";
import mammoth from "mammoth";
import XLSX from "xlsx";
import JSZip from "jszip";
import { FirestoreSessionStore, type StudyBuddyState } from "./firestore-session-store.js";
import { fromThreadMessages, toGenkitMessages, toThreadMessages, type ChatHistoryMessage } from "./chat-history.js";
import {
  asOptionalString,
  asRequiredString,
  badRequest,
  okJson,
  sendErrorResponse,
  sendServerError,
  setSSEHeaders,
} from "./http-utils.js";

initializeApp();
const db = getFirestore();
const storage = getStorage();
const MAIN_THREAD = "main";

const DEFAULT_MODEL_NAME = (process.env.OPENAI_MODEL || "gpt-4o-mini").trim() || "gpt-4o-mini";
const MODEL_NAME_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{1,100}$/;
const FALLBACK_ALLOWED_MODELS = [
  "gpt-4o-mini",
  "gpt-4o",
  "gpt-4.1-nano",
  "gpt-4.1-mini",
  "gpt-4.1",
  "gpt-5-nano",
  "gpt-5-mini",
  "gpt-5",
  "gpt-5.1",
  "gpt-5.2",
  "gpt-5.4",
];

function parseModelNameList(raw: string | undefined): string[] {
  if (!raw) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of raw.split(",")) {
    const modelName = token.trim();
    if (!modelName || seen.has(modelName)) continue;
    seen.add(modelName);
    out.push(modelName);
  }
  return out;
}

function uniqueModelNames(candidates: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const candidate of candidates) {
    if (!candidate || seen.has(candidate)) continue;
    seen.add(candidate);
    out.push(candidate);
  }
  return out;
}

const configuredAllowedModels = parseModelNameList(process.env.OPENAI_ALLOWED_MODELS);
const ALLOWED_CHAT_MODEL_NAMES = configuredAllowedModels.length > 0
  ? uniqueModelNames([DEFAULT_MODEL_NAME, ...configuredAllowedModels])
  : uniqueModelNames([DEFAULT_MODEL_NAME, ...FALLBACK_ALLOWED_MODELS]);
const ALLOWED_CHAT_MODEL_SET = new Set(ALLOWED_CHAT_MODEL_NAMES);

function getChatModel(modelName: string) {
  return openAI.model(modelName);
}

function resolveRequestedChatModelName(value: unknown): string {
  if (typeof value !== "string") return DEFAULT_MODEL_NAME;
  const requested = value.trim();
  if (!requested) return DEFAULT_MODEL_NAME;
  if (!MODEL_NAME_PATTERN.test(requested)) return DEFAULT_MODEL_NAME;
  if (!ALLOWED_CHAT_MODEL_SET.has(requested)) return DEFAULT_MODEL_NAME;
  return requested;
}

const CHAT_MODEL = getChatModel(DEFAULT_MODEL_NAME);
const OCR_MODEL_NAME = (process.env.OPENAI_OCR_MODEL || DEFAULT_MODEL_NAME).trim() || DEFAULT_MODEL_NAME;

const ai = genkit({
  plugins: [openAI({
    apiKey: process.env.OPENAI_API_KEY || "",
  })],
  model: CHAT_MODEL,
});

const sessionStore = new FirestoreSessionStore<StudyBuddyState>(db, MAIN_THREAD);

type StudyCoachMode = "nudge" | "recap";
type MaterialFileType = "pdf" | "docx" | "spreadsheet" | "slides" | "txt" | "image";
type MaterialLocationType = "page" | "sheet" | "slide" | "line" | "image";
type StudyDifficulty = "easy" | "medium" | "hard";
type StudySourceType = "chat" | "material";
type StudyQuizQuestionType = "mcq" | "short";
type FlashcardReviewRating = "again" | "hard" | "good" | "easy";
type FocusSessionMode = "study_mode" | "exam_simulation";
type ExamSimulationStatus = "generating" | "ready" | "in_progress" | "completed" | "timed_out" | "abandoned" | "failed";
type ExamConfidence = "low" | "medium" | "high";
type GamificationBadgeId =
  | "first_focus_day"
  | "streak_3"
  | "streak_7"
  | "streak_14"
  | "streak_30"
  | "weekly_goal_1"
  | "focus_300m"
  | "focus_1000m";

type FocusSessionDoc = {
  id: string;
  userId: string;
  status: string;
  startedAt: Date | null;
  endedAt: Date | null;
  mode?: FocusSessionMode | null;
  chatId?: string | null;
  examSimulationId?: string | null;
};

type FocusSummaryDoc = {
  userId: string;
  focusedMs: number;
  distractedMs: number;
  focusPercent: number;
  endTs: number | null;
  createdAt: Date | null;
  distractions?: number;
  firstDriftOffsetSec?: number | null;
};

type GamificationProfileDoc = {
  id: string;
  userId: string;
  currentStreakDays: number;
  longestStreakDays: number;
  lastQualifiedDayKey: string | null;
  totalXp: number;
  level: number;
  totalFocusedMinutes: number;
  weeklyGoalTargetMinutes: number;
  currentWeekKey: string;
  currentWeekStartDayKey: string;
  currentWeekFocusedMinutes: number;
  currentWeekCompletedAt: Date | null;
  unlockedBadges: GamificationBadgeId[];
  createdAt: Date;
  updatedAt: Date;
};

type GamificationSessionAwardDoc = {
  id: string;
  focusSessionId: string;
  userId: string;
  dayKey: string;
  weekKey: string;
  weekStartDayKey: string;
  timezone: string;
  focusedMinutes: number;
  baseXp: number;
  qualityMultiplier: number;
  xpGain: number;
  focusPercent: number;
  badgeUnlocks: GamificationBadgeId[];
  createdAt: Date | null;
};

type CourseMaterialDoc = {
  id: string;
  userId: string;
  courseId?: string | null;
  sessionId?: string | null;
  chatId?: string | null;
  fileName: string;
  extension?: string | null;
  mimeType?: string | null;
  storagePath: string;
  fileType?: MaterialFileType;
  status?: string;
};

type MaterialChunkDoc = {
  id: string;
  materialId: string;
  userId: string;
  courseId: string | null;
  sessionId: string | null;
  chatId: string | null;
  fileName: string;
  fileType: MaterialFileType;
  locationType: MaterialLocationType;
  locationLabel: string;
  chunkIndex: number;
  text: string;
  textLower: string;
  createdAt: Date;
};

type ExtractedSegment = {
  text: string;
  locationType: MaterialLocationType;
  locationLabel: string;
};

type RagCitation = {
  id: string;
  materialId: string;
  fileName: string;
  fileType: MaterialFileType;
  locationType: MaterialLocationType;
  locationLabel: string;
  snippet: string;
  score: number;
  contextText: string;
};

type StudySourceDoc = {
  id: string;
  type: StudySourceType;
  label: string;
  snippet: string;
};

type StudyQuizQuestionDoc = {
  id: string;
  questionType: StudyQuizQuestionType;
  prompt: string;
  options: string[];
  correctAnswer: string;
  correctOptionIndex: number | null;
  explanation: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

type StudyFlashcardDoc = {
  id: string;
  front: string;
  back: string;
  tags: string[];
  difficulty: StudyDifficulty;
  sourceIds: string[];
  nextReviewAt: Date;
  intervalDays: number;
  easeFactor: number;
  repetitions: number;
  lastReviewedAt: Date | null;
};

type StudyExamQuestionDoc = {
  id: string;
  prompt: string;
  rubric: string[];
  modelAnswer: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

type StudySetDoc = {
  id: string;
  userId: string;
  chatId: string;
  courseId: string | null;
  sessionId: string | null;
  status: "generating" | "ready" | "failed";
  quizQuestions: StudyQuizQuestionDoc[];
  flashcards: StudyFlashcardDoc[];
  examQuestions: StudyExamQuestionDoc[];
  sources: StudySourceDoc[];
  model: string;
  generationMs: number | null;
  errorMessage: string | null;
  createdAt: Date;
  updatedAt: Date;
};

type ExamSimulationSectionDoc = {
  id: string;
  label: string;
  questionTargetCount: number;
  targetDurationSec: number;
  cumulativeTargetSec: number;
};

type ExamSimulationQuestionDoc = {
  id: string;
  prompt: string;
  options: string[];
  difficulty: StudyDifficulty;
  topic: string;
  sourceIds: string[];
  sectionId: string;
  orderIndex: number;
};

type ExamSimulationResponseDoc = {
  questionId: string;
  questionOrder: number;
  selectedOptionIndex: number;
  confidence: ExamConfidence;
  elapsedSec: number;
  answeredAt: Date;
  difficulty: StudyDifficulty;
  topic: string;
  sectionId: string;
};

type ExamSimulationWeakTopicDoc = {
  topic: string;
  accuracyPercent: number;
  questionCount: number;
  correctCount: number;
};

type ExamSimulationDifficultyBreakdownDoc = {
  difficulty: StudyDifficulty;
  accuracyPercent: number;
  questionCount: number;
  correctCount: number;
};

type ExamSimulationSectionResultDoc = {
  sectionId: string;
  label: string;
  targetDurationSec: number;
  actualElapsedSec: number;
  overrunSec: number;
  answeredCount: number;
};

type ExamSimulationFocusInsightDoc = {
  focusSessionId: string;
  focusPercent: number;
  distractions: number;
  firstDriftOffsetSec: number | null;
};

type ExamSimulationRecapDoc = {
  answeredCount: number;
  totalQuestionCount: number;
  correctCount: number;
  scorePercent: number;
  completionPercent: number;
  overconfidenceMisses: number;
  weakTopics: ExamSimulationWeakTopicDoc[];
  accuracyByDifficulty: ExamSimulationDifficultyBreakdownDoc[];
  sectionResults: ExamSimulationSectionResultDoc[];
  timeLossMoments: string[];
  recoveryPlan: string[];
  weakTopicSummary: string;
  focusInsight: ExamSimulationFocusInsightDoc | null;
};

type ExamSimulationDoc = {
  id: string;
  userId: string;
  chatId: string;
  courseId: string | null;
  sessionId: string | null;
  status: ExamSimulationStatus;
  preset: "standard_mock";
  durationSec: number;
  servedQuestionCount: number;
  questionBankCount: number;
  model: string;
  sections: ExamSimulationSectionDoc[];
  servedQuestions: ExamSimulationQuestionDoc[];
  responses: ExamSimulationResponseDoc[];
  currentQuestionId: string | null;
  recap: ExamSimulationRecapDoc | null;
  errorMessage: string | null;
  createdAt: Date;
  updatedAt: Date;
  startedAt: Date | null;
  endsAt: Date | null;
  finishedAt: Date | null;
};

type ExamSimulationQuestionStateDoc = {
  id: string;
  prompt: string;
  options: string[];
  correctOptionIndex: number;
  correctOption: string;
  explanation: string;
  difficulty: StudyDifficulty;
  topic: string;
  sourceIds: string[];
};

type ExamSimulationInternalResponseDoc = ExamSimulationResponseDoc & {
  isCorrect: boolean;
};

type ExamSimulationStateDoc = {
  id: string;
  examSimulationId: string;
  userId: string;
  status: ExamSimulationStatus;
  currentDifficulty: StudyDifficulty;
  questionBank: ExamSimulationQuestionStateDoc[];
  allSources: StudySourceDoc[];
  remainingQuestionIds: string[];
  servedQuestionIds: string[];
  currentQuestionId: string | null;
  internalResponses: ExamSimulationInternalResponseDoc[];
  createdAt: Date;
  updatedAt: Date;
  startedAt: Date | null;
  endsAt: Date | null;
  finishedAt: Date | null;
};

const SYSTEM_INSTRUCTION = `You are an AI Study Buddy - a knowledgeable, patient, and encouraging learning companion designed to help students succeed academically.

Your core principles:
- Be friendly, approachable, and supportive in all interactions
- Break down complex concepts into clear, digestible explanations
- Use examples and analogies to make learning more relatable
- Encourage questions and create a safe learning environment
- Adapt your teaching style to the student's level and needs
- Provide step-by-step guidance when explaining processes or solving problems
- Acknowledge effort and progress to build confidence
- Be concise but thorough - avoid overwhelming with too much information at once

When answering questions:
- Start with a clear, direct answer
- Then provide context and deeper explanation if needed
- Use numbered lists or bullet points for multi-step processes
- Include relevant examples or real-world applications
- Ask follow-up questions to check understanding when appropriate

Remember: Your goal is to help students learn effectively, not just provide answers. Foster critical thinking and independent learning skills.`;

const FUNCTION_CONFIG = {
  cors: true,
  region: "us-central1" as const,
  secrets: ["OPENAI_API_KEY"],
};

const MAX_SOURCE_FILE_SIZE_BYTES = 25 * 1024 * 1024;
const MAX_RETRIEVAL_CHUNKS = 6;
const CHUNK_MAX_LEN = 1100;
const CHUNK_OVERLAP = 140;
const MAX_CHAT_MESSAGES_FOR_STUDY_SET = 30;
const MAX_STUDY_TRANSCRIPT_CHARS = 14000;
const DEFAULT_QUIZ_COUNT = 10;
const DEFAULT_FLASHCARD_COUNT = 14;
const DEFAULT_EXAM_COUNT = 4;
const EXAM_SIMULATION_DURATION_SEC = 30 * 60;
const EXAM_SIMULATION_SERVED_QUESTION_COUNT = 10;
const EXAM_SIMULATION_BANK_COUNT = 15;
const EXAM_SIMULATION_MIN_SOURCE_CHAR_COUNT = 240;
const EXAM_SIMULATION_SECTION_SPECS: ExamSimulationSectionDoc[] = [
  {
    id: "section_1",
    label: "Section 1",
    questionTargetCount: 3,
    targetDurationSec: 10 * 60,
    cumulativeTargetSec: 10 * 60,
  },
  {
    id: "section_2",
    label: "Section 2",
    questionTargetCount: 3,
    targetDurationSec: 10 * 60,
    cumulativeTargetSec: 20 * 60,
  },
  {
    id: "section_3",
    label: "Section 3",
    questionTargetCount: 4,
    targetDurationSec: 10 * 60,
    cumulativeTargetSec: 30 * 60,
  },
];
const DAILY_STREAK_TARGET_MINUTES = 25;
const WEEKLY_GOAL_TARGET_MINUTES = 180;
const XP_PER_LEVEL = 100;
const MAX_FOCUSED_MINUTES_PER_SESSION = 12 * 60;
const GAMIFICATION_BADGE_IDS: GamificationBadgeId[] = [
  "first_focus_day",
  "streak_3",
  "streak_7",
  "streak_14",
  "streak_30",
  "weekly_goal_1",
  "focus_300m",
  "focus_1000m",
];

const SUPPORTED_EXTENSIONS = new Set(["pdf", "docx", "xlsx", "xls", "pptx", "ppt", "txt", "png", "jpg", "jpeg", "webp"]);

const STOPWORDS = new Set([
  "the", "and", "for", "with", "that", "this", "from", "have", "what", "when", "where", "which", "into", "your", "you",
  "about", "does", "are", "was", "were", "then", "than", "there", "their", "them", "been", "can", "could", "would",
  "should", "how", "why", "who", "all", "any", "each", "our", "its", "not", "but", "use", "using", "used",
]);

function asRequiredNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asOptionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clampInteger(input: number | null, fallback: number, min: number, max: number): number {
  if (input == null) return fallback;
  const floored = Math.floor(input);
  if (!Number.isFinite(floored)) return fallback;
  return Math.min(max, Math.max(min, floored));
}

function asDate(value: unknown): Date | null {
  if (value instanceof Date) return value;
  if (value && typeof value === "object" && "toDate" in value) {
    try {
      const out = (value as { toDate?: () => Date }).toDate?.();
      if (out instanceof Date) return out;
    } catch {
      return null;
    }
  }
  return null;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function normalizeTimeZone(value: string | null): string {
  const candidate = value?.trim();
  if (!candidate) return "UTC";
  try {
    // Throws if IANA timezone is invalid.
    new Intl.DateTimeFormat("en-US", { timeZone: candidate }).format(new Date());
    return candidate;
  } catch {
    return "UTC";
  }
}

function formatDayKey(year: number, month: number, day: number): string {
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function parseDayKey(dayKey: string): { year: number; month: number; day: number } | null {
  const match = dayKey.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  const year = Number.parseInt(match[1], 10);
  const month = Number.parseInt(match[2], 10);
  const day = Number.parseInt(match[3], 10);
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > 31) return null;
  return { year, month, day };
}

function dayKeyToUtcDate(dayKey: string): Date | null {
  const parsed = parseDayKey(dayKey);
  if (!parsed) return null;
  return new Date(Date.UTC(parsed.year, parsed.month - 1, parsed.day));
}

function isNextDayKey(previousDayKey: string, nextDayKey: string): boolean {
  const previous = dayKeyToUtcDate(previousDayKey);
  const next = dayKeyToUtcDate(nextDayKey);
  if (!previous || !next) return false;
  const diffDays = Math.round((next.getTime() - previous.getTime()) / (24 * 60 * 60 * 1000));
  return diffDays === 1;
}

function getDayKeyForTimeZone(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);

  const year = Number.parseInt(parts.find((part) => part.type === "year")?.value || "0", 10);
  const month = Number.parseInt(parts.find((part) => part.type === "month")?.value || "0", 10);
  const day = Number.parseInt(parts.find((part) => part.type === "day")?.value || "0", 10);

  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day) || year <= 0 || month <= 0 || day <= 0) {
    const fallback = dayKeyToUtcDate(formatDayKey(date.getUTCFullYear(), date.getUTCMonth() + 1, date.getUTCDate()));
    if (!fallback) {
      return "1970-01-01";
    }
    return formatDayKey(fallback.getUTCFullYear(), fallback.getUTCMonth() + 1, fallback.getUTCDate());
  }

  return formatDayKey(year, month, day);
}

function getWeekInfoFromDayKey(dayKey: string): { weekKey: string; weekStartDayKey: string } {
  const dayDate = dayKeyToUtcDate(dayKey);
  if (!dayDate) {
    return { weekKey: "1970-W01", weekStartDayKey: "1970-01-01" };
  }

  const isoDay = dayDate.getUTCDay() === 0 ? 7 : dayDate.getUTCDay();
  const weekStartDate = new Date(dayDate.getTime());
  weekStartDate.setUTCDate(weekStartDate.getUTCDate() - (isoDay - 1));
  const weekStartDayKey = formatDayKey(
    weekStartDate.getUTCFullYear(),
    weekStartDate.getUTCMonth() + 1,
    weekStartDate.getUTCDate(),
  );

  const thursday = new Date(dayDate.getTime());
  thursday.setUTCDate(thursday.getUTCDate() + (4 - isoDay));
  const weekYear = thursday.getUTCFullYear();
  const yearStart = new Date(Date.UTC(weekYear, 0, 1));
  const weekNo = Math.ceil((((thursday.getTime() - yearStart.getTime()) / (24 * 60 * 60 * 1000)) + 1) / 7);
  const weekKey = `${weekYear}-W${String(weekNo).padStart(2, "0")}`;

  return { weekKey, weekStartDayKey };
}

function getWeekInfoForDate(date: Date, timeZone: string): { dayKey: string; weekKey: string; weekStartDayKey: string } {
  const dayKey = getDayKeyForTimeZone(date, timeZone);
  const weekInfo = getWeekInfoFromDayKey(dayKey);
  return { dayKey, weekKey: weekInfo.weekKey, weekStartDayKey: weekInfo.weekStartDayKey };
}

function getQualityMultiplier(focusPercent: number): number {
  if (focusPercent >= 85) return 1.25;
  if (focusPercent >= 70) return 1.1;
  return 1.0;
}

function getLevelFromTotalXp(totalXp: number): number {
  return Math.floor(Math.max(0, totalXp) / XP_PER_LEVEL) + 1;
}

function isGamificationBadgeId(value: string): value is GamificationBadgeId {
  return (GAMIFICATION_BADGE_IDS as string[]).includes(value);
}

function normalizeBadgeIds(value: unknown): GamificationBadgeId[] {
  if (!Array.isArray(value)) return [];
  const out: GamificationBadgeId[] = [];
  const seen = new Set<GamificationBadgeId>();
  for (const row of value) {
    const badgeId = asString(row);
    if (!badgeId || !isGamificationBadgeId(badgeId)) continue;
    if (seen.has(badgeId)) continue;
    seen.add(badgeId);
    out.push(badgeId);
  }
  return out;
}

function asTimestampMs(value: unknown): number | null {
  const date = asDate(value);
  if (date) return date.getTime();
  const numeric = asOptionalNumber(value);
  return numeric == null ? null : numeric;
}

function parseFocusSessionDoc(docId: string, data: Record<string, unknown>): FocusSessionDoc {
  return {
    id: docId,
    userId: asString(data.userId),
    status: asString(data.status),
    startedAt: asDate(data.startedAt),
    endedAt: asDate(data.endedAt),
    mode: normalizeFocusSessionMode(data.mode),
    chatId: asString(data.chatId) || null,
    examSimulationId: asString(data.examSimulationId) || null,
  };
}

function parseFocusSummaryDoc(data: Record<string, unknown>): FocusSummaryDoc {
  const focusedMs = Math.max(0, asOptionalNumber(data.focusedMs) ?? 0);
  const distractedMs = Math.max(0, asOptionalNumber(data.distractedMs) ?? 0);
  const computedFocusPercent = focusedMs + distractedMs > 0 ? (focusedMs / (focusedMs + distractedMs)) * 100 : 0;
  const focusPercent = clampNumber(asOptionalNumber(data.focusPercent) ?? computedFocusPercent, 0, 100);
  return {
    userId: asString(data.userId),
    focusedMs,
    distractedMs,
    focusPercent: Math.round(focusPercent * 10) / 10,
    endTs: asOptionalNumber(data.endTs),
    createdAt: asDate(data.createdAt),
    distractions: Math.max(0, clampInteger(asOptionalNumber(data.distractions), 0, 0, 1000000)),
    firstDriftOffsetSec: asOptionalNumber(data.firstDriftOffsetSec),
  };
}

function parseGamificationProfileDoc(
  userId: string,
  data: Record<string, unknown> | null,
  now: Date,
  fallbackWeekKey: string,
  fallbackWeekStartDayKey: string,
): GamificationProfileDoc {
  if (!data) {
    return {
      id: userId,
      userId,
      currentStreakDays: 0,
      longestStreakDays: 0,
      lastQualifiedDayKey: null,
      totalXp: 0,
      level: 1,
      totalFocusedMinutes: 0,
      weeklyGoalTargetMinutes: WEEKLY_GOAL_TARGET_MINUTES,
      currentWeekKey: fallbackWeekKey,
      currentWeekStartDayKey: fallbackWeekStartDayKey,
      currentWeekFocusedMinutes: 0,
      currentWeekCompletedAt: null,
      unlockedBadges: [],
      createdAt: now,
      updatedAt: now,
    };
  }

  return {
    id: userId,
    userId,
    currentStreakDays: Math.max(0, clampInteger(asOptionalNumber(data.currentStreakDays), 0, 0, 100000)),
    longestStreakDays: Math.max(0, clampInteger(asOptionalNumber(data.longestStreakDays), 0, 0, 100000)),
    lastQualifiedDayKey: asString(data.lastQualifiedDayKey) || null,
    totalXp: Math.max(0, clampInteger(asOptionalNumber(data.totalXp), 0, 0, 100000000)),
    level: Math.max(1, clampInteger(asOptionalNumber(data.level), 1, 1, 1000000)),
    totalFocusedMinutes: Math.max(0, clampInteger(asOptionalNumber(data.totalFocusedMinutes), 0, 0, 100000000)),
    weeklyGoalTargetMinutes: Math.max(1, clampInteger(asOptionalNumber(data.weeklyGoalTargetMinutes), WEEKLY_GOAL_TARGET_MINUTES, 1, 1000000)),
    currentWeekKey: asString(data.currentWeekKey) || fallbackWeekKey,
    currentWeekStartDayKey: asString(data.currentWeekStartDayKey) || fallbackWeekStartDayKey,
    currentWeekFocusedMinutes: Math.max(0, clampInteger(asOptionalNumber(data.currentWeekFocusedMinutes), 0, 0, 1000000)),
    currentWeekCompletedAt: asDate(data.currentWeekCompletedAt),
    unlockedBadges: normalizeBadgeIds(data.unlockedBadges),
    createdAt: asDate(data.createdAt) || now,
    updatedAt: asDate(data.updatedAt) || now,
  };
}

function parseSessionAwardDoc(focusSessionId: string, userId: string, data: Record<string, unknown>): GamificationSessionAwardDoc {
  return {
    id: focusSessionId,
    focusSessionId,
    userId: asString(data.userId) || userId,
    dayKey: asString(data.dayKey),
    weekKey: asString(data.weekKey),
    weekStartDayKey: asString(data.weekStartDayKey),
    timezone: asString(data.timezone) || "UTC",
    focusedMinutes: Math.max(0, clampInteger(asOptionalNumber(data.focusedMinutes), 0, 0, 1000000)),
    baseXp: Math.max(0, clampInteger(asOptionalNumber(data.baseXp), 0, 0, 1000000)),
    qualityMultiplier: clampNumber(asOptionalNumber(data.qualityMultiplier) ?? 1, 1, 2),
    xpGain: Math.max(0, clampInteger(asOptionalNumber(data.xpGain), 0, 0, 10000000)),
    focusPercent: clampNumber(asOptionalNumber(data.focusPercent) ?? 0, 0, 100),
    badgeUnlocks: normalizeBadgeIds(data.badgeUnlocks),
    createdAt: asDate(data.createdAt),
  };
}

function toGamificationAwardResponse(award: GamificationSessionAwardDoc): {
  focusSessionId: string;
  dayKey: string;
  weekKey: string;
  focusedMinutes: number;
  baseXp: number;
  qualityMultiplier: number;
  xpGain: number;
  focusPercent: number;
  badgeUnlocks: GamificationBadgeId[];
} {
  return {
    focusSessionId: award.focusSessionId,
    dayKey: award.dayKey,
    weekKey: award.weekKey,
    focusedMinutes: award.focusedMinutes,
    baseXp: award.baseXp,
    qualityMultiplier: award.qualityMultiplier,
    xpGain: award.xpGain,
    focusPercent: award.focusPercent,
    badgeUnlocks: award.badgeUnlocks,
  };
}

function toGamificationProfileResponse(profile: GamificationProfileDoc): {
  currentStreakDays: number;
  longestStreakDays: number;
  lastQualifiedDayKey: string | null;
  totalXp: number;
  level: number;
  levelProgressXp: number;
  xpToNextLevel: number;
  totalFocusedMinutes: number;
  weeklyGoal: {
    targetMinutes: number;
    weekKey: string;
    weekStartDayKey: string;
    focusedMinutes: number;
    completedAt: string | null;
  };
  unlockedBadges: GamificationBadgeId[];
} {
  const levelProgressXp = profile.totalXp % XP_PER_LEVEL;
  const xpToNextLevel = XP_PER_LEVEL - levelProgressXp;
  return {
    currentStreakDays: profile.currentStreakDays,
    longestStreakDays: profile.longestStreakDays,
    lastQualifiedDayKey: profile.lastQualifiedDayKey,
    totalXp: profile.totalXp,
    level: profile.level,
    levelProgressXp,
    xpToNextLevel: xpToNextLevel === 0 ? XP_PER_LEVEL : xpToNextLevel,
    totalFocusedMinutes: profile.totalFocusedMinutes,
    weeklyGoal: {
      targetMinutes: profile.weeklyGoalTargetMinutes,
      weekKey: profile.currentWeekKey,
      weekStartDayKey: profile.currentWeekStartDayKey,
      focusedMinutes: profile.currentWeekFocusedMinutes,
      completedAt: profile.currentWeekCompletedAt ? profile.currentWeekCompletedAt.toISOString() : null,
    },
    unlockedBadges: profile.unlockedBadges,
  };
}

function normalizeDifficulty(value: unknown): StudyDifficulty {
  const candidate = asString(value).toLowerCase();
  if (candidate === "easy" || candidate === "medium" || candidate === "hard") return candidate;
  return "medium";
}

function normalizeFocusSessionMode(value: unknown): FocusSessionMode | null {
  const candidate = asString(value).toLowerCase();
  if (candidate === "study_mode" || candidate === "exam_simulation") return candidate;
  return null;
}

function normalizeExamConfidence(value: unknown): ExamConfidence | null {
  const candidate = asString(value).toLowerCase();
  if (candidate === "low" || candidate === "medium" || candidate === "high") return candidate;
  return null;
}

function cloneExamSections(): ExamSimulationSectionDoc[] {
  return EXAM_SIMULATION_SECTION_SPECS.map((section) => ({ ...section }));
}

function getExamSectionForOrder(orderIndex: number): ExamSimulationSectionDoc {
  let cursor = 0;
  for (const section of EXAM_SIMULATION_SECTION_SPECS) {
    cursor += section.questionTargetCount;
    if (orderIndex < cursor) {
      return section;
    }
  }
  return EXAM_SIMULATION_SECTION_SPECS[EXAM_SIMULATION_SECTION_SPECS.length - 1];
}

function getHarderDifficulty(difficulty: StudyDifficulty): StudyDifficulty {
  if (difficulty === "easy") return "medium";
  if (difficulty === "medium") return "hard";
  return "hard";
}

function getEasierDifficulty(difficulty: StudyDifficulty): StudyDifficulty {
  if (difficulty === "hard") return "medium";
  if (difficulty === "medium") return "easy";
  return "easy";
}

function getExamFallbackDifficultyOrder(target: StudyDifficulty): StudyDifficulty[] {
  if (target === "easy") return ["easy", "medium", "hard"];
  if (target === "hard") return ["hard", "medium", "easy"];
  return ["medium", "hard", "easy"];
}

function normalizeSourceIds(value: unknown, allowed: Set<string>): string[] {
  if (!Array.isArray(value)) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const row of value) {
    const id = asString(row);
    if (!id || !allowed.has(id) || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
    if (out.length >= 4) break;
  }
  return out;
}

function extractJsonText(raw: string): string {
  const trimmed = raw.trim();
  const fenceMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenceMatch && typeof fenceMatch[1] === "string") {
    return fenceMatch[1].trim();
  }
  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }
  return trimmed;
}

function parseJsonObject(raw: string): Record<string, unknown> {
  const jsonText = extractJsonText(raw);
  const parsed = JSON.parse(jsonText);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Study generation response must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

function createHttpError(statusCode: number, message: string): Error & { statusCode: number } {
  const error = new Error(message) as Error & { statusCode: number };
  error.statusCode = statusCode;
  return error;
}

function getFileExtension(fileName: string | null | undefined): string {
  if (!fileName) return "";
  const idx = fileName.lastIndexOf(".");
  if (idx < 0 || idx === fileName.length - 1) return "";
  return fileName.slice(idx + 1).toLowerCase();
}

function inferFileType(fileName: string, mimeType: string | null | undefined): MaterialFileType | null {
  const ext = getFileExtension(fileName);
  if (ext === "pdf" || mimeType === "application/pdf") return "pdf";
  if (ext === "docx" || mimeType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") return "docx";
  if (ext === "xlsx" || ext === "xls" || (mimeType || "").includes("spreadsheet") || mimeType === "application/vnd.ms-excel") return "spreadsheet";
  if (ext === "pptx" || ext === "ppt" || (mimeType || "").includes("presentation") || mimeType === "application/vnd.ms-powerpoint") return "slides";
  if (ext === "txt" || mimeType === "text/plain") return "txt";
  if (["png", "jpg", "jpeg", "webp"].includes(ext) || (mimeType || "").startsWith("image/")) return "image";
  return null;
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\r/g, "\n").replace(/\t/g, " ").replace(/[ \u00a0]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function truncateText(text: string, maxLength: number): string {
  const normalized = text.trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 1)}...`;
}

function decodeXmlEntities(input: string): string {
  return input
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_m, code: string) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_m, hex: string) => String.fromCharCode(Number.parseInt(hex, 16)));
}

function extractTextFromSlideXml(xml: string): string {
  const parts: string[] = [];
  const pattern = /<a:t[^>]*>([\s\S]*?)<\/a:t>/g;
  let match: RegExpExecArray | null = pattern.exec(xml);
  while (match) {
    parts.push(decodeXmlEntities(match[1]));
    match = pattern.exec(xml);
  }
  return normalizeWhitespace(parts.join("\n"));
}

function extractSlideNumber(path: string): number {
  const found = path.match(/slide(\d+)\.xml$/i);
  if (!found) return Number.MAX_SAFE_INTEGER;
  return Number.parseInt(found[1], 10);
}

async function extractTextFromImageWithOpenAI(buffer: Buffer, mimeType: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY || "";
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured for OCR");
  }

  const payload = {
    model: OCR_MODEL_NAME,
    messages: [
      {
        role: "system",
        content: "You extract text from images for study materials. Return plain text only. Do not add commentary.",
      },
      {
        role: "user",
        content: [
          { type: "text", text: "Extract all readable text from this image. Keep original wording as best as possible." },
          {
            type: "image_url",
            image_url: {
              url: `data:${mimeType};base64,${buffer.toString("base64")}`,
            },
          },
        ],
      },
    ],
    temperature: 0,
    max_tokens: 1200,
  };

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Image OCR failed (${response.status}): ${text || "unknown error"}`);
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | Array<{ type?: string; text?: string }> } }>;
  };

  const content = data.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return normalizeWhitespace(content);
  }
  if (Array.isArray(content)) {
    const txt = content
      .map((part) => (part?.type === "text" && typeof part.text === "string" ? part.text : ""))
      .join("\n");
    return normalizeWhitespace(txt);
  }

  return "";
}

async function extractSegmentsFromMaterial(buffer: Buffer, material: CourseMaterialDoc): Promise<{ fileType: MaterialFileType; segments: ExtractedSegment[] }> {
  const inferred = inferFileType(material.fileName, material.mimeType);
  if (!inferred) {
    throw new Error(`Unsupported material type: ${material.fileName}`);
  }

  if (!SUPPORTED_EXTENSIONS.has(getFileExtension(material.fileName))) {
    // Keep MIME-based support but reject unknown extensions for safety.
    if (!["pdf", "docx", "spreadsheet", "slides", "txt", "image"].includes(inferred)) {
      throw new Error(`Unsupported file extension for ${material.fileName}`);
    }
  }

  if (inferred === "pdf") {
    const parser = new PDFParse({ data: buffer });
    const parsed = await parser.getText();
    await parser.destroy();
    const rawText = typeof parsed.text === "string" ? parsed.text : "";
    const pages = rawText
      .split(/\f+/)
      .map((text: string) => normalizeWhitespace(text))
      .filter((text: string) => text.length > 0);

    if (pages.length > 1) {
      return {
        fileType: "pdf",
        segments: pages.map((text: string, idx: number) => ({
          text,
          locationType: "page",
          locationLabel: `Page ${idx + 1}`,
        })),
      };
    }

    const text = normalizeWhitespace(rawText);
    if (!text) throw new Error("No readable text found in PDF");
    return {
      fileType: "pdf",
      segments: [{ text, locationType: "page", locationLabel: "Page 1" }],
    };
  }

  if (inferred === "docx") {
    const result = await mammoth.extractRawText({ buffer });
    const text = normalizeWhitespace(result.value || "");
    if (!text) throw new Error("No readable text found in DOCX");
    return {
      fileType: "docx",
      segments: [{ text, locationType: "line", locationLabel: "Document" }],
    };
  }

  if (inferred === "spreadsheet") {
    const workbook = XLSX.read(buffer, { type: "buffer" });
    const segments: ExtractedSegment[] = [];

    for (const sheetName of workbook.SheetNames) {
      const sheet = workbook.Sheets[sheetName];
      if (!sheet) continue;

      const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, { header: 1, raw: false, defval: "" });
      const lines: string[] = [];
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex] || [];
        const values = row.map((cell) => String(cell).trim()).filter((cell) => cell.length > 0);
        if (values.length === 0) continue;
        lines.push(`Row ${rowIndex + 1}: ${values.join(" | ")}`);
      }

      const text = normalizeWhitespace(lines.join("\n"));
      if (text.length > 0) {
        segments.push({
          text,
          locationType: "sheet",
          locationLabel: `Sheet ${sheetName}`,
        });
      }
    }

    if (segments.length === 0) throw new Error("No readable text found in spreadsheet");
    return { fileType: "spreadsheet", segments };
  }

  if (inferred === "slides") {
    const zip = await JSZip.loadAsync(buffer);
    const slidePaths = Object.keys(zip.files)
      .filter((path) => /^ppt\/slides\/slide\d+\.xml$/i.test(path))
      .sort((a, b) => extractSlideNumber(a) - extractSlideNumber(b));

    const segments: ExtractedSegment[] = [];
    for (const slidePath of slidePaths) {
      // eslint-disable-next-line no-await-in-loop
      const slideXml = await zip.file(slidePath)?.async("string");
      if (!slideXml) continue;
      const text = extractTextFromSlideXml(slideXml);
      if (!text) continue;
      segments.push({
        text,
        locationType: "slide",
        locationLabel: `Slide ${extractSlideNumber(slidePath)}`,
      });
    }

    if (segments.length === 0) throw new Error("No readable text found in slide deck");
    return { fileType: "slides", segments };
  }

  if (inferred === "txt") {
    const text = normalizeWhitespace(buffer.toString("utf8"));
    if (!text) throw new Error("No readable text found in text file");
    return {
      fileType: "txt",
      segments: [{ text, locationType: "line", locationLabel: "Text" }],
    };
  }

  const mimeType = material.mimeType || "image/png";
  const text = await extractTextFromImageWithOpenAI(buffer, mimeType);
  if (!text) {
    throw new Error("No readable text found in image");
  }
  return {
    fileType: "image",
    segments: [{ text, locationType: "image", locationLabel: "Image OCR" }],
  };
}

function splitTextIntoChunks(text: string, maxLen: number, overlap: number): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return [];
  if (normalized.length <= maxLen) return [normalized];

  const chunks: string[] = [];
  let start = 0;

  while (start < normalized.length) {
    let end = Math.min(start + maxLen, normalized.length);

    if (end < normalized.length) {
      const minBoundary = Math.floor(start + maxLen * 0.6);
      const window = normalized.slice(minBoundary, end);
      const boundaryOffset = Math.max(window.lastIndexOf("\n"), window.lastIndexOf("."), window.lastIndexOf(" "));
      if (boundaryOffset > 0) {
        end = minBoundary + boundaryOffset + 1;
      }
    }

    const chunk = normalizeWhitespace(normalized.slice(start, end));
    if (chunk) {
      chunks.push(chunk);
    }

    if (end >= normalized.length) break;
    const nextStart = Math.max(0, end - overlap);
    if (nextStart <= start) {
      start = end;
    } else {
      start = nextStart;
    }
  }

  return chunks;
}

function buildChunkDocs(material: CourseMaterialDoc, fileType: MaterialFileType, segments: ExtractedSegment[]): MaterialChunkDoc[] {
  const out: MaterialChunkDoc[] = [];
  let chunkIndex = 0;

  for (const segment of segments) {
    const segmentChunks = splitTextIntoChunks(segment.text, CHUNK_MAX_LEN, CHUNK_OVERLAP);
    for (const text of segmentChunks) {
      chunkIndex += 1;
      const id = `${material.id}_${String(chunkIndex).padStart(5, "0")}`;
      out.push({
        id,
        materialId: material.id,
        userId: material.userId,
        courseId: material.courseId ?? null,
        sessionId: material.sessionId ?? null,
        chatId: material.chatId ?? null,
        fileName: material.fileName,
        fileType,
        locationType: segment.locationType,
        locationLabel: segment.locationLabel,
        chunkIndex,
        text,
        textLower: text.toLowerCase(),
        createdAt: new Date(),
      });
    }
  }

  return out;
}

async function deleteMaterialChunks(dbRef: Firestore, materialId: string): Promise<void> {
  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const snapshot = await dbRef.collection("material_chunks").where("materialId", "==", materialId).limit(400).get();
    if (snapshot.empty) return;

    const batch = dbRef.batch();
    for (const doc of snapshot.docs) {
      batch.delete(doc.ref);
    }
    // eslint-disable-next-line no-await-in-loop
    await batch.commit();

    if (snapshot.size < 400) return;
  }
}

async function writeMaterialChunks(dbRef: Firestore, chunks: MaterialChunkDoc[]): Promise<void> {
  if (chunks.length === 0) return;

  for (let i = 0; i < chunks.length; i += 400) {
    const batch = dbRef.batch();
    const slice = chunks.slice(i, i + 400);
    for (const chunk of slice) {
      batch.set(dbRef.collection("material_chunks").doc(chunk.id), chunk);
    }
    // eslint-disable-next-line no-await-in-loop
    await batch.commit();
  }
}

function tokenize(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !STOPWORDS.has(token));
}

function scoreChunk(queryTerms: Set<string>, chunk: MaterialChunkDoc): number {
  if (queryTerms.size === 0) return 0;
  const chunkTokens = tokenize(chunk.textLower);
  if (chunkTokens.length === 0) return 0;

  let score = 0;
  const seen = new Set<string>();
  for (const token of chunkTokens) {
    if (!queryTerms.has(token)) continue;
    score += 1;
    if (!seen.has(token)) {
      seen.add(token);
      score += 0.6;
    }
  }

  return score;
}

async function retrieveRankedCitations(chatId: string, userId: string, userQuery: string): Promise<RagCitation[]> {
  const snapshot = await db.collection("material_chunks").where("chatId", "==", chatId).limit(1200).get();
  if (snapshot.empty) return [];

  const chunks = snapshot.docs
    .map((doc) => doc.data() as MaterialChunkDoc)
    .filter((chunk) => chunk.userId === userId && typeof chunk.text === "string" && chunk.text.trim().length > 0);

  if (chunks.length === 0) return [];

  const terms = new Set(tokenize(userQuery));
  const scored = chunks
    .map((chunk) => ({ chunk, score: scoreChunk(terms, chunk) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_RETRIEVAL_CHUNKS);

  if (scored.length === 0) return [];

  return scored.map((item, idx) => ({
    id: `C${idx + 1}`,
    materialId: item.chunk.materialId,
    fileName: item.chunk.fileName,
    fileType: item.chunk.fileType,
    locationType: item.chunk.locationType,
    locationLabel: item.chunk.locationLabel,
    snippet: truncateText(item.chunk.text, 260),
    score: Math.round(item.score * 100) / 100,
    contextText: truncateText(item.chunk.text, 900),
  }));
}

function buildRagUserPrompt(question: string, citations: RagCitation[]): string {
  if (citations.length === 0) return question;

  const context = citations
    .map((citation) => {
      return [
        `[${citation.id}] ${citation.fileName} (${citation.locationLabel})`,
        citation.contextText,
      ].join("\n");
    })
    .join("\n\n");

  return [
    "You may use the context snippets below as optional supporting references.",
    "Rules:",
    "- Answer normally; general knowledge is allowed.",
    "- Use snippets when they are relevant to the user's question.",
    "- Cite snippet-backed claims inline with [C#] markers.",
    "- Do not invent citations or claim a snippet says something it does not.",
    "- If the user explicitly asks about uploaded material and needed info is missing, say that clearly.",
    "",
    "Context snippets:",
    context,
    "",
    `Question: ${question}`,
  ].join("\n");
}

function selectFinalCitations(answerText: string, citations: RagCitation[]): RagCitation[] {
  if (citations.length === 0) return [];
  const used = new Set<string>();
  const regex = /\[(C\d+)\]/g;
  let match: RegExpExecArray | null = regex.exec(answerText);
  while (match) {
    used.add(match[1]);
    match = regex.exec(answerText);
  }

  if (used.size === 0) return [];

  return citations.filter((citation) => used.has(citation.id));
}

type ChatMessageForStudy = {
  id: string;
  text: string;
  isAI: boolean;
  createdAt: Date | null;
};

async function loadChatOwnership(chatId: string): Promise<{ userId: string; courseId: string | null; sessionId: string | null } | null> {
  const chatRef = db.collection("chats").doc(chatId);
  const chatDoc = await chatRef.get();
  if (!chatDoc.exists) return null;
  const data = chatDoc.data() as Record<string, unknown>;
  return {
    userId: typeof data.userId === "string" ? data.userId : "",
    courseId: typeof data.courseId === "string" ? data.courseId : null,
    sessionId: typeof data.sessionId === "string" ? data.sessionId : null,
  };
}

async function loadRecentChatMessages(chatId: string, userId: string, limitCount: number): Promise<ChatMessageForStudy[]> {
  const snapshot = await db.collection("messages")
    .where("sessionId", "==", chatId)
    .where("userId", "==", userId)
    .orderBy("createdAt", "desc")
    .limit(limitCount)
    .get();

  return snapshot.docs
    .map((doc) => {
      const data = doc.data() as Record<string, unknown>;
      const textField = typeof data.text === "string"
        ? data.text
        : (data.text && typeof data.text === "object" && "text" in data.text)
          ? asString((data.text as Record<string, unknown>).text)
          : "";
      return {
        id: doc.id,
        text: textField,
        isAI: !!data.isAI,
        createdAt: asDate(data.createdAt),
      } satisfies ChatMessageForStudy;
    })
    .sort((a, b) => {
      const aTime = a.createdAt ? a.createdAt.getTime() : 0;
      const bTime = b.createdAt ? b.createdAt.getTime() : 0;
      return aTime - bTime;
    });
}

function buildChatSourcesAndTranscript(messages: ChatMessageForStudy[]): { sources: StudySourceDoc[]; transcript: string } {
  const sources: StudySourceDoc[] = [];
  const lines: string[] = [];
  let idx = 0;

  for (const message of messages) {
    const normalizedText = normalizeWhitespace(message.text || "");
    if (!normalizedText) continue;
    idx += 1;
    const sourceId = `M${idx}`;
    const role = message.isAI ? "AI" : "You";
    sources.push({
      id: sourceId,
      type: "chat",
      label: `${role} message ${idx}`,
      snippet: truncateText(normalizedText, 220),
    });
    lines.push(`[${sourceId}] ${role}: ${truncateText(normalizedText, 420)}`);
  }

  let transcript = lines.join("\n");
  if (transcript.length > MAX_STUDY_TRANSCRIPT_CHARS) {
    transcript = transcript.slice(transcript.length - MAX_STUDY_TRANSCRIPT_CHARS);
  }

  return { sources, transcript };
}

function buildMaterialSources(citations: RagCitation[]): StudySourceDoc[] {
  return citations.map((citation) => ({
    id: citation.id,
    type: "material",
    label: `${citation.fileName} (${citation.locationLabel})`,
    snippet: truncateText(citation.contextText || citation.snippet, 220),
  }));
}

function buildStudyGenerationPrompt({
  quizCount,
  flashcardCount,
  examCount,
  transcript,
  sources,
}: {
  quizCount: number;
  flashcardCount: number;
  examCount: number;
  transcript: string;
  sources: StudySourceDoc[];
}): string {
  const sourceCatalog = sources
    .map((source) => `- ${source.id} [${source.type}] ${source.label}: ${source.snippet}`)
    .join("\n");

  return [
    "Generate active-recall study assets from the chat transcript and source catalog.",
    "Output only valid JSON (no markdown, no prose) with this top-level shape:",
    "{",
    '  "quiz": [',
    "    {",
    '      "type": "mcq" | "short",',
    '      "question": "string",',
    '      "choices": ["string"]  // required for mcq only, 4 options preferred,',
    '      "answerIndex": 0,        // required for mcq only, 0-based index into choices,',
    '      "answer": "string",     // required for short only,',
    '      "explanation": "string",',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["M1","C1"]',
    "    }",
    "  ],",
    '  "flashcards": [',
    "    {",
    '      "front": "string",',
    '      "back": "string",',
    '      "tags": ["string"],',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["M2","C2"]',
    "    }",
    "  ],",
    '  "examQuestions": [',
    "    {",
    '      "prompt": "string",',
    '      "rubric": ["criterion string"],',
    '      "modelAnswer": "string",',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["C1","M3"]',
    "    }",
    "  ]",
    "}",
    "",
    `Required counts: quiz=${quizCount}, flashcards=${flashcardCount}, examQuestions=${examCount}.`,
    "Rules:",
    "- Use only facts supported by transcript/source catalog.",
    "- Every item must include at least one source id from the catalog.",
    "- Mix difficulty levels across the set.",
    "- Keep wording concise and student-friendly.",
    "",
    "Source catalog:",
    sourceCatalog || "- none",
    "",
    "Chat transcript:",
    transcript || "(empty)",
  ].join("\n");
}

function parseStudyQuizQuestions(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
): StudyQuizQuestionDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyQuizQuestionDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const questionType: StudyQuizQuestionType = asString(data.type).toLowerCase() === "mcq" ? "mcq" : "short";
    const prompt = asString(data.question) || asString(data.prompt);
    if (prompt.length < 6) continue;

    const explanation = asString(data.explanation);
    const difficulty = normalizeDifficulty(data.difficulty);
    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    if (questionType === "mcq") {
      const options = Array.isArray(data.choices)
        ? data.choices.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 6)
        : [];
      if (options.length < 2) continue;

      let correctOptionIndex: number | null =
        typeof data.answerIndex === "number" && Number.isFinite(data.answerIndex)
          ? Math.floor(data.answerIndex)
          : null;
      if (correctOptionIndex == null || correctOptionIndex < 0 || correctOptionIndex >= options.length) {
        const answerText = asString(data.answer).toLowerCase();
        const matchIndex = answerText ? options.findIndex((option) => option.toLowerCase() === answerText) : -1;
        correctOptionIndex = matchIndex >= 0 ? matchIndex : 0;
      }

      out.push({
        id: `Q${out.length + 1}`,
        questionType: "mcq",
        prompt,
        options,
        correctAnswer: options[correctOptionIndex] || options[0],
        correctOptionIndex,
        explanation,
        difficulty,
        sourceIds: normalizedSourceIds,
      });
      continue;
    }

    const shortAnswer = asString(data.answer) || asString(data.correctAnswer);
    if (!shortAnswer) continue;
    out.push({
      id: `Q${out.length + 1}`,
      questionType: "short",
      prompt,
      options: [],
      correctAnswer: shortAnswer,
      correctOptionIndex: null,
      explanation,
      difficulty,
      sourceIds: normalizedSourceIds,
    });
  }

  return out;
}

function parseStudyFlashcards(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
  now: Date,
): StudyFlashcardDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyFlashcardDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const front = asString(data.front);
    const back = asString(data.back);
    if (front.length < 4 || back.length < 4) continue;

    const tags = Array.isArray(data.tags)
      ? data.tags.map((item) => asString(item)).filter((tag) => tag.length > 0).slice(0, 6)
      : [];

    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    out.push({
      id: `F${out.length + 1}`,
      front,
      back,
      tags,
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: normalizedSourceIds,
      nextReviewAt: new Date(now.getTime() + 24 * 60 * 60 * 1000),
      intervalDays: 1,
      easeFactor: 2.5,
      repetitions: 0,
      lastReviewedAt: null,
    });
  }

  return out;
}

function parseStudyExamQuestions(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
): StudyExamQuestionDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyExamQuestionDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const prompt = asString(data.prompt) || asString(data.question);
    if (prompt.length < 6) continue;

    const rubric = Array.isArray(data.rubric)
      ? data.rubric.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 6)
      : [];
    const modelAnswer = asString(data.modelAnswer) || asString(data.answer);
    if (!modelAnswer) continue;

    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    out.push({
      id: `E${out.length + 1}`,
      prompt,
      rubric,
      modelAnswer,
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: normalizedSourceIds,
    });
  }

  return out;
}

function buildExamSimulationPrompt({
  transcript,
  sources,
}: {
  transcript: string;
  sources: StudySourceDoc[];
}): string {
  const sourceCatalog = sources
    .map((source) => `- ${source.id} [${source.type}] ${source.label}: ${source.snippet}`)
    .join("\n");

  return [
    "Generate a source-grounded mock exam question bank.",
    "Output only valid JSON (no markdown, no prose) with this exact top-level shape:",
    "{",
    '  "questions": [',
    "    {",
    '      "prompt": "string",',
    '      "choices": ["string", "string", "string", "string"],',
    '      "answerIndex": 0,',
    '      "explanation": "string",',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "topic": "string",',
    '      "sourceIds": ["M1","C1"]',
    "    }",
    "  ]",
    "}",
    "",
    `Return exactly ${EXAM_SIMULATION_BANK_COUNT} questions total.`,
    "Return exactly 5 easy, 5 medium, and 5 hard questions.",
    "All questions must be MCQ with exactly 4 choices.",
    "Every question must include at least one valid source id from the source catalog.",
    "Keep prompts concise, exam-like, and non-trivial.",
    "Avoid duplicate prompts or near-duplicate answer choices.",
    "Use only facts supported by the transcript and source catalog.",
    "",
    "Source catalog:",
    sourceCatalog || "- none",
    "",
    "Chat transcript:",
    transcript || "(empty)",
  ].join("\n");
}

function parseExamSimulationQuestionBank(
  value: unknown,
  allowedSourceIds: Set<string>,
): ExamSimulationQuestionStateDoc[] {
  if (!Array.isArray(value)) return [];
  const out: ExamSimulationQuestionStateDoc[] = [];
  const seenPrompts = new Set<string>();
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= EXAM_SIMULATION_BANK_COUNT) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const prompt = asString(data.prompt) || asString(data.question);
    const promptKey = normalizeWhitespace(prompt).toLowerCase();
    if (prompt.length < 8 || seenPrompts.has(promptKey)) continue;

    const options = Array.isArray(data.choices)
      ? data.choices.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 4)
      : [];
    if (options.length !== 4) continue;

    let correctOptionIndex =
      typeof data.answerIndex === "number" && Number.isFinite(data.answerIndex)
        ? Math.floor(data.answerIndex)
        : -1;
    if (correctOptionIndex < 0 || correctOptionIndex >= options.length) {
      const answerText = asString(data.answer).toLowerCase();
      const matchIndex = answerText ? options.findIndex((option) => option.toLowerCase() === answerText) : -1;
      correctOptionIndex = matchIndex >= 0 ? matchIndex : -1;
    }
    if (correctOptionIndex < 0 || correctOptionIndex >= options.length) continue;

    const difficulty = normalizeDifficulty(data.difficulty);
    const topic = asString(data.topic) || `${difficulty} review`;
    const explanation = asString(data.explanation) || "Review the cited material and compare each option carefully.";
    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);
    if (normalizedSourceIds.length === 0) continue;

    seenPrompts.add(promptKey);
    out.push({
      id: `B${out.length + 1}`,
      prompt,
      options,
      correctOptionIndex,
      correctOption: options[correctOptionIndex] || "",
      explanation,
      difficulty,
      topic,
      sourceIds: normalizedSourceIds,
    });
  }

  return out;
}

function hasExpectedExamDifficultyDistribution(questionBank: ExamSimulationQuestionStateDoc[]): boolean {
  if (questionBank.length !== EXAM_SIMULATION_BANK_COUNT) return false;
  const counts: Record<StudyDifficulty, number> = {
    easy: 0,
    medium: 0,
    hard: 0,
  };
  for (const question of questionBank) {
    counts[question.difficulty] += 1;
  }
  return counts.easy === 5 && counts.medium === 5 && counts.hard === 5;
}

function determineNextExamTargetDifficulty(
  currentDifficulty: StudyDifficulty,
  isCorrect: boolean,
  confidence: ExamConfidence,
): StudyDifficulty {
  if (isCorrect && confidence === "high") {
    return getHarderDifficulty(currentDifficulty);
  }
  if (!isCorrect && (confidence === "medium" || confidence === "high")) {
    return getEasierDifficulty(currentDifficulty);
  }
  return currentDifficulty;
}

function pickNextExamQuestion(
  questionBank: ExamSimulationQuestionStateDoc[],
  remainingQuestionIds: string[],
  targetDifficulty: StudyDifficulty,
): ExamSimulationQuestionStateDoc | null {
  const remaining = new Set(remainingQuestionIds);
  for (const difficulty of getExamFallbackDifficultyOrder(targetDifficulty)) {
    const match = questionBank.find((question) => remaining.has(question.id) && question.difficulty === difficulty);
    if (match) return match;
  }
  return null;
}

function toPublicExamQuestion(
  question: ExamSimulationQuestionStateDoc,
  orderIndex: number,
): ExamSimulationQuestionDoc {
  const section = getExamSectionForOrder(orderIndex);
  return {
    id: question.id,
    prompt: question.prompt,
    options: [...question.options],
    difficulty: question.difficulty,
    topic: question.topic,
    sourceIds: [...question.sourceIds],
    sectionId: section.id,
    orderIndex,
  };
}

function formatDurationLabel(totalSec: number): string {
  const minutes = Math.max(0, Math.floor(totalSec / 60));
  const seconds = Math.max(0, totalSec % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function roundPercent(value: number): number {
  return Math.round(clampNumber(value, 0, 100) * 10) / 10;
}

function buildFallbackWeakTopicSummary(weakTopics: ExamSimulationWeakTopicDoc[], answeredCount: number): string {
  if (answeredCount === 0) {
    return "No graded answers yet, so the main opportunity is to complete a full mock exam under time pressure.";
  }
  if (weakTopics.length === 0) {
    return "Your accuracy stayed steady across topics, so the next gain is likely pacing rather than raw content review.";
  }
  if (weakTopics.length === 1) {
    return `Your biggest gap was ${weakTopics[0].topic}, where accuracy landed at ${weakTopics[0].accuracyPercent}%.`;
  }
  return `Your weakest topics were ${weakTopics[0].topic} and ${weakTopics[1].topic}, where accuracy dropped most under exam pacing.`;
}

function buildFallbackRecoveryPlan(args: {
  weakTopics: ExamSimulationWeakTopicDoc[];
  timeLossMoments: string[];
  overconfidenceMisses: number;
  focusInsight: ExamSimulationFocusInsightDoc | null;
}): string[] {
  const primaryTopic = args.weakTopics[0]?.topic || "the missed concepts from this run";
  const recoveryPlan = [
    `Review ${primaryTopic} first and rebuild the core rule or concept before attempting another timed set.`,
    args.overconfidenceMisses > 0
      ? "Slow down on medium and hard questions where you feel certain quickly, and force a brief elimination check before locking an answer."
      : "Keep confidence honest by marking unsure questions early and deciding faster when you know the answer.",
    args.timeLossMoments.length > 0
      ? "Run one more 30-minute mock within 48 hours and focus on staying inside section pace, not just improving raw accuracy."
      : "Run one more 30-minute mock within 48 hours to confirm the same topics stay stable under time pressure.",
  ];
  if (args.focusInsight && args.focusInsight.firstDriftOffsetSec != null) {
    recoveryPlan[2] = `Run one more 30-minute mock within 48 hours and reset your attention before ${formatDurationLabel(args.focusInsight.firstDriftOffsetSec)} if you feel drift starting.`;
  }
  return recoveryPlan.slice(0, 3);
}

function buildExamRecoveryPrompt(args: {
  weakTopics: ExamSimulationWeakTopicDoc[];
  accuracyByDifficulty: ExamSimulationDifficultyBreakdownDoc[];
  sectionResults: ExamSimulationSectionResultDoc[];
  timeLossMoments: string[];
  overconfidenceMisses: number;
  focusInsight: ExamSimulationFocusInsightDoc | null;
  allSources: StudySourceDoc[];
}): string {
  const sourceCatalog = args.allSources
    .slice(0, 12)
    .map((source) => `- ${source.id} [${source.type}] ${source.label}`)
    .join("\n");

  return [
    "You are generating a post-exam recovery summary.",
    "Use only the supplied metrics and topics. Do not invent topics, scores, or time events.",
    "Return valid JSON only with this shape:",
    "{",
    '  "weakTopicSummary": "string",',
    '  "timeLossMoments": ["string"],',
    '  "recoveryPlan": ["string", "string", "string"]',
    "}",
    "",
    `Weak topics: ${JSON.stringify(args.weakTopics)}`,
    `Accuracy by difficulty: ${JSON.stringify(args.accuracyByDifficulty)}`,
    `Section pacing: ${JSON.stringify(args.sectionResults)}`,
    `Time-loss moments: ${JSON.stringify(args.timeLossMoments)}`,
    `Overconfidence misses: ${args.overconfidenceMisses}`,
    `Focus insight: ${JSON.stringify(args.focusInsight)}`,
    "",
    "Keep the summary concise, actionable, and student-friendly.",
    "The recovery plan must contain exactly 3 steps.",
    "",
    "Source labels for topic naming context:",
    sourceCatalog || "- none",
  ].join("\n");
}

async function loadExamFocusInsight(
  focusSessionId: string | null,
  userId: string,
): Promise<ExamSimulationFocusInsightDoc | null> {
  if (!focusSessionId) return null;
  const snapshot = await db.collection("focusSummaries").doc(focusSessionId).get();
  if (!snapshot.exists) return null;
  const summary = parseFocusSummaryDoc((snapshot.data() || {}) as Record<string, unknown>);
  if (summary.userId !== userId) return null;
  return {
    focusSessionId,
    focusPercent: roundPercent(summary.focusPercent),
    distractions: Math.max(0, summary.distractions ?? 0),
    firstDriftOffsetSec: summary.firstDriftOffsetSec ?? null,
  };
}

function buildExamWeakTopics(responses: ExamSimulationInternalResponseDoc[]): ExamSimulationWeakTopicDoc[] {
  const topicMap = new Map<string, { total: number; correct: number }>();
  for (const response of responses) {
    const existing = topicMap.get(response.topic) || { total: 0, correct: 0 };
    existing.total += 1;
    if (response.isCorrect) existing.correct += 1;
    topicMap.set(response.topic, existing);
  }

  return [...topicMap.entries()]
    .map(([topic, stats]) => ({
      topic,
      questionCount: stats.total,
      correctCount: stats.correct,
      accuracyPercent: stats.total > 0 ? roundPercent((stats.correct / stats.total) * 100) : 0,
    }))
    .sort((a, b) => {
      if (a.accuracyPercent !== b.accuracyPercent) return a.accuracyPercent - b.accuracyPercent;
      return b.questionCount - a.questionCount;
    })
    .slice(0, 3);
}

function buildExamDifficultyBreakdown(responses: ExamSimulationInternalResponseDoc[]): ExamSimulationDifficultyBreakdownDoc[] {
  const difficulties: StudyDifficulty[] = ["easy", "medium", "hard"];
  return difficulties.map((difficulty) => {
    const filtered = responses.filter((response) => response.difficulty === difficulty);
    const correctCount = filtered.filter((response) => response.isCorrect).length;
    return {
      difficulty,
      questionCount: filtered.length,
      correctCount,
      accuracyPercent: filtered.length > 0 ? roundPercent((correctCount / filtered.length) * 100) : 0,
    };
  });
}

function buildExamSectionResults(
  responses: ExamSimulationInternalResponseDoc[],
  sections: ExamSimulationSectionDoc[],
): ExamSimulationSectionResultDoc[] {
  const sorted = [...responses].sort((a, b) => a.questionOrder - b.questionOrder);
  let previousElapsedSec = 0;
  return sections.map((section) => {
    const sectionResponses = sorted.filter((response) => response.sectionId === section.id);
    const maxElapsedSec = sectionResponses.reduce((max, response) => Math.max(max, response.elapsedSec), previousElapsedSec);
    const actualElapsedSec = Math.max(0, maxElapsedSec - previousElapsedSec);
    previousElapsedSec = maxElapsedSec;
    return {
      sectionId: section.id,
      label: section.label,
      targetDurationSec: section.targetDurationSec,
      actualElapsedSec,
      overrunSec: Math.max(0, actualElapsedSec - section.targetDurationSec),
      answeredCount: sectionResponses.length,
    };
  });
}

function buildExamTimeLossMoments(args: {
  responses: ExamSimulationInternalResponseDoc[];
  sectionResults: ExamSimulationSectionResultDoc[];
  focusInsight: ExamSimulationFocusInsightDoc | null;
}): string[] {
  const moments: string[] = [];
  for (const sectionResult of args.sectionResults) {
    if (sectionResult.overrunSec <= 0) continue;
    moments.push(
      `${sectionResult.label} ran ${formatDurationLabel(sectionResult.overrunSec)} over pace.`
    );
  }

  const sortedResponses = [...args.responses].sort((a, b) => a.questionOrder - b.questionOrder);
  let previousElapsedSec = 0;
  const slowQuestions: Array<{ questionOrder: number; deltaSec: number; targetSec: number }> = [];
  for (const response of sortedResponses) {
    const section = EXAM_SIMULATION_SECTION_SPECS.find((item) => item.id === response.sectionId) || EXAM_SIMULATION_SECTION_SPECS[0];
    const targetPerQuestionSec = Math.max(1, Math.round(section.targetDurationSec / section.questionTargetCount));
    const deltaSec = Math.max(0, response.elapsedSec - previousElapsedSec);
    previousElapsedSec = response.elapsedSec;
    if (deltaSec > targetPerQuestionSec * 1.35) {
      slowQuestions.push({
        questionOrder: response.questionOrder,
        deltaSec,
        targetSec: targetPerQuestionSec,
      });
    }
  }

  slowQuestions
    .sort((a, b) => b.deltaSec - a.deltaSec)
    .slice(0, 2)
    .forEach((item) => {
      moments.push(
        `Question ${item.questionOrder + 1} used ${formatDurationLabel(item.deltaSec)} against a ${formatDurationLabel(item.targetSec)} target.`
      );
    });

  if (args.focusInsight) {
    if (args.focusInsight.firstDriftOffsetSec != null) {
      moments.push(
        `Focus drift started around ${formatDurationLabel(args.focusInsight.firstDriftOffsetSec)}.`
      );
    }
    if (args.focusInsight.distractions > 0) {
      moments.push(
        `${args.focusInsight.distractions} distraction event${args.focusInsight.distractions === 1 ? "" : "s"} were detected during the run.`
      );
    }
  }

  return moments.slice(0, 5);
}

async function buildExamSimulationRecap(args: {
  userId: string;
  modelName: string;
  sections: ExamSimulationSectionDoc[];
  allSources: StudySourceDoc[];
  responses: ExamSimulationInternalResponseDoc[];
  focusSessionId: string | null;
}): Promise<ExamSimulationRecapDoc> {
  const answeredCount = args.responses.length;
  const correctCount = args.responses.filter((response) => response.isCorrect).length;
  const weakTopics = buildExamWeakTopics(args.responses);
  const accuracyByDifficulty = buildExamDifficultyBreakdown(args.responses);
  const sectionResults = buildExamSectionResults(args.responses, args.sections);
  const focusInsight = await loadExamFocusInsight(args.focusSessionId, args.userId);
  const overconfidenceMisses = args.responses.filter(
    (response) => !response.isCorrect && response.confidence === "high",
  ).length;
  const fallbackTimeLossMoments = buildExamTimeLossMoments({
    responses: args.responses,
    sectionResults,
    focusInsight,
  });
  const fallbackWeakTopicSummary = buildFallbackWeakTopicSummary(weakTopics, answeredCount);
  const fallbackRecoveryPlan = buildFallbackRecoveryPlan({
    weakTopics,
    timeLossMoments: fallbackTimeLossMoments,
    overconfidenceMisses,
    focusInsight,
  });

  let weakTopicSummary = fallbackWeakTopicSummary;
  let timeLossMoments = fallbackTimeLossMoments;
  let recoveryPlan = fallbackRecoveryPlan;

  try {
    const selectedChatModel = getChatModel(resolveRequestedChatModelName(args.modelName));
    const generation: GenerateResponse = await ai.generate({
      model: selectedChatModel,
      system: "You turn exam performance metrics into a concise study recovery plan. Return JSON only.",
      prompt: buildExamRecoveryPrompt({
        weakTopics,
        accuracyByDifficulty,
        sectionResults,
        timeLossMoments: fallbackTimeLossMoments,
        overconfidenceMisses,
        focusInsight,
        allSources: args.allSources,
      }),
      config: {
        temperature: 0.25,
        maxOutputTokens: 900,
      },
    });

    const parsed = parseJsonObject(generation.text || "");
    const parsedWeakTopicSummary = asString(parsed.weakTopicSummary);
    const parsedTimeLossMoments = Array.isArray(parsed.timeLossMoments)
      ? parsed.timeLossMoments.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 5)
      : [];
    const parsedRecoveryPlan = Array.isArray(parsed.recoveryPlan)
      ? parsed.recoveryPlan.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 3)
      : [];

    if (parsedWeakTopicSummary) {
      weakTopicSummary = parsedWeakTopicSummary;
    }
    if (parsedTimeLossMoments.length > 0) {
      timeLossMoments = parsedTimeLossMoments;
    }
    if (parsedRecoveryPlan.length === 3) {
      recoveryPlan = parsedRecoveryPlan;
    }
  } catch (error) {
    console.warn("Exam recovery plan generation failed, using fallback recap.", error);
  }

  return {
    answeredCount,
    totalQuestionCount: EXAM_SIMULATION_SERVED_QUESTION_COUNT,
    correctCount,
    scorePercent: answeredCount > 0 ? roundPercent((correctCount / answeredCount) * 100) : 0,
    completionPercent: roundPercent((answeredCount / EXAM_SIMULATION_SERVED_QUESTION_COUNT) * 100),
    overconfidenceMisses,
    weakTopics,
    accuracyByDifficulty,
    sectionResults,
    timeLossMoments,
    recoveryPlan,
    weakTopicSummary,
    focusInsight,
  };
}

function normalizeExamSimulationStatus(value: unknown): ExamSimulationStatus {
  const candidate = asString(value).toLowerCase();
  if (
    candidate === "generating" ||
    candidate === "ready" ||
    candidate === "in_progress" ||
    candidate === "completed" ||
    candidate === "timed_out" ||
    candidate === "abandoned" ||
    candidate === "failed"
  ) {
    return candidate;
  }
  return "ready";
}

function parseExamSimulationSections(value: unknown): ExamSimulationSectionDoc[] {
  if (!Array.isArray(value)) return cloneExamSections();
  const sections = value
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const data = row as Record<string, unknown>;
      const id = asString(data.id);
      const label = asString(data.label);
      const questionTargetCount = clampInteger(asOptionalNumber(data.questionTargetCount), 0, 0, 100);
      const targetDurationSec = clampInteger(asOptionalNumber(data.targetDurationSec), 0, 0, 100000);
      const cumulativeTargetSec = clampInteger(asOptionalNumber(data.cumulativeTargetSec), 0, 0, 100000);
      if (!id || !label || questionTargetCount <= 0 || targetDurationSec <= 0 || cumulativeTargetSec <= 0) return null;
      return {
        id,
        label,
        questionTargetCount,
        targetDurationSec,
        cumulativeTargetSec,
      } satisfies ExamSimulationSectionDoc;
    })
    .filter((section): section is ExamSimulationSectionDoc => section !== null);

  return sections.length > 0 ? sections : cloneExamSections();
}

function parseExamSimulationQuestions(value: unknown): ExamSimulationQuestionDoc[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const data = row as Record<string, unknown>;
      const id = asString(data.id);
      const prompt = asString(data.prompt);
      const sectionId = asString(data.sectionId);
      if (!id || !prompt || !sectionId) return null;
      const options = Array.isArray(data.options)
        ? data.options.map((item) => asString(item)).filter((item) => item.length > 0)
        : [];
      if (options.length === 0) return null;
      const sourceIds = Array.isArray(data.sourceIds)
        ? data.sourceIds.map((item) => asString(item)).filter((item) => item.length > 0)
        : [];
      return {
        id,
        prompt,
        options,
        difficulty: normalizeDifficulty(data.difficulty),
        topic: asString(data.topic) || "General review",
        sourceIds,
        sectionId,
        orderIndex: clampInteger(asOptionalNumber(data.orderIndex), 0, 0, 100000),
      } satisfies ExamSimulationQuestionDoc;
    })
    .filter((question): question is ExamSimulationQuestionDoc => question !== null)
    .sort((a, b) => a.orderIndex - b.orderIndex);
}

function parseExamSimulationResponses(value: unknown): ExamSimulationResponseDoc[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const data = row as Record<string, unknown>;
      const questionId = asString(data.questionId);
      const sectionId = asString(data.sectionId);
      const confidence = normalizeExamConfidence(data.confidence);
      if (!questionId || !sectionId || !confidence) return null;
      return {
        questionId,
        questionOrder: clampInteger(asOptionalNumber(data.questionOrder), 0, 0, 100000),
        selectedOptionIndex: clampInteger(asOptionalNumber(data.selectedOptionIndex), 0, 0, 100),
        confidence,
        elapsedSec: clampInteger(asOptionalNumber(data.elapsedSec), 0, 0, 100000),
        answeredAt: asDate(data.answeredAt) || new Date(0),
        difficulty: normalizeDifficulty(data.difficulty),
        topic: asString(data.topic) || "General review",
        sectionId,
      } satisfies ExamSimulationResponseDoc;
    })
    .filter((response): response is ExamSimulationResponseDoc => response !== null)
    .sort((a, b) => a.questionOrder - b.questionOrder);
}

function parseExamSimulationRecap(value: unknown): ExamSimulationRecapDoc | null {
  if (!value || typeof value !== "object") return null;
  const data = value as Record<string, unknown>;

  const weakTopics = Array.isArray(data.weakTopics)
    ? data.weakTopics
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          const topic = asString(item.topic);
          if (!topic) return null;
          return {
            topic,
            accuracyPercent: roundPercent(asOptionalNumber(item.accuracyPercent) ?? 0),
            questionCount: clampInteger(asOptionalNumber(item.questionCount), 0, 0, 1000),
            correctCount: clampInteger(asOptionalNumber(item.correctCount), 0, 0, 1000),
          } satisfies ExamSimulationWeakTopicDoc;
        })
        .filter((item): item is ExamSimulationWeakTopicDoc => item !== null)
    : [];

  const accuracyByDifficulty = Array.isArray(data.accuracyByDifficulty)
    ? data.accuracyByDifficulty
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          return {
            difficulty: normalizeDifficulty(item.difficulty),
            accuracyPercent: roundPercent(asOptionalNumber(item.accuracyPercent) ?? 0),
            questionCount: clampInteger(asOptionalNumber(item.questionCount), 0, 0, 1000),
            correctCount: clampInteger(asOptionalNumber(item.correctCount), 0, 0, 1000),
          } satisfies ExamSimulationDifficultyBreakdownDoc;
        })
        .filter((item): item is ExamSimulationDifficultyBreakdownDoc => item !== null)
    : [];

  const sectionResults = Array.isArray(data.sectionResults)
    ? data.sectionResults
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          const sectionId = asString(item.sectionId);
          const label = asString(item.label);
          if (!sectionId || !label) return null;
          return {
            sectionId,
            label,
            targetDurationSec: clampInteger(asOptionalNumber(item.targetDurationSec), 0, 0, 100000),
            actualElapsedSec: clampInteger(asOptionalNumber(item.actualElapsedSec), 0, 0, 100000),
            overrunSec: clampInteger(asOptionalNumber(item.overrunSec), 0, 0, 100000),
            answeredCount: clampInteger(asOptionalNumber(item.answeredCount), 0, 0, 1000),
          } satisfies ExamSimulationSectionResultDoc;
        })
        .filter((item): item is ExamSimulationSectionResultDoc => item !== null)
    : [];

  const timeLossMoments = Array.isArray(data.timeLossMoments)
    ? data.timeLossMoments.map((item) => asString(item)).filter((item) => item.length > 0)
    : [];
  const recoveryPlan = Array.isArray(data.recoveryPlan)
    ? data.recoveryPlan.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 3)
    : [];

  const focusInsight =
    data.focusInsight && typeof data.focusInsight === "object"
      ? (() => {
          const insight = data.focusInsight as Record<string, unknown>;
          const focusSessionId = asString(insight.focusSessionId);
          if (!focusSessionId) return null;
          return {
            focusSessionId,
            focusPercent: roundPercent(asOptionalNumber(insight.focusPercent) ?? 0),
            distractions: clampInteger(asOptionalNumber(insight.distractions), 0, 0, 100000),
            firstDriftOffsetSec: asOptionalNumber(insight.firstDriftOffsetSec),
          } satisfies ExamSimulationFocusInsightDoc;
        })()
      : null;

  return {
    answeredCount: clampInteger(asOptionalNumber(data.answeredCount), 0, 0, 1000),
    totalQuestionCount: clampInteger(asOptionalNumber(data.totalQuestionCount), EXAM_SIMULATION_SERVED_QUESTION_COUNT, 0, 1000),
    correctCount: clampInteger(asOptionalNumber(data.correctCount), 0, 0, 1000),
    scorePercent: roundPercent(asOptionalNumber(data.scorePercent) ?? 0),
    completionPercent: roundPercent(asOptionalNumber(data.completionPercent) ?? 0),
    overconfidenceMisses: clampInteger(asOptionalNumber(data.overconfidenceMisses), 0, 0, 1000),
    weakTopics,
    accuracyByDifficulty,
    sectionResults,
    timeLossMoments,
    recoveryPlan,
    weakTopicSummary: asString(data.weakTopicSummary),
    focusInsight,
  };
}

function parseExamSimulationDoc(docId: string, data: Record<string, unknown>): ExamSimulationDoc {
  return {
    id: docId,
    userId: asString(data.userId),
    chatId: asString(data.chatId),
    courseId: asString(data.courseId) || null,
    sessionId: asString(data.sessionId) || null,
    status: normalizeExamSimulationStatus(data.status),
    preset: "standard_mock",
    durationSec: clampInteger(asOptionalNumber(data.durationSec), EXAM_SIMULATION_DURATION_SEC, 1, 100000),
    servedQuestionCount: clampInteger(asOptionalNumber(data.servedQuestionCount), EXAM_SIMULATION_SERVED_QUESTION_COUNT, 1, 1000),
    questionBankCount: clampInteger(asOptionalNumber(data.questionBankCount), EXAM_SIMULATION_BANK_COUNT, 1, 1000),
    model: asString(data.model) || DEFAULT_MODEL_NAME,
    sections: parseExamSimulationSections(data.sections),
    servedQuestions: parseExamSimulationQuestions(data.servedQuestions),
    responses: parseExamSimulationResponses(data.responses),
    currentQuestionId: asString(data.currentQuestionId) || null,
    recap: parseExamSimulationRecap(data.recap),
    errorMessage: asString(data.errorMessage) || null,
    createdAt: asDate(data.createdAt) || new Date(0),
    updatedAt: asDate(data.updatedAt) || new Date(0),
    startedAt: asDate(data.startedAt),
    endsAt: asDate(data.endsAt),
    finishedAt: asDate(data.finishedAt),
  };
}

function parseExamSimulationStateDoc(examSimulationId: string, data: Record<string, unknown>): ExamSimulationStateDoc {
  const questionBank = Array.isArray(data.questionBank)
    ? data.questionBank
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          const id = asString(item.id);
          const prompt = asString(item.prompt);
          if (!id || !prompt) return null;
          const options = Array.isArray(item.options)
            ? item.options.map((entry) => asString(entry)).filter((entry) => entry.length > 0)
            : [];
          if (options.length === 0) return null;
          const sourceIds = Array.isArray(item.sourceIds)
            ? item.sourceIds.map((entry) => asString(entry)).filter((entry) => entry.length > 0)
            : [];
          return {
            id,
            prompt,
            options,
            correctOptionIndex: clampInteger(asOptionalNumber(item.correctOptionIndex), 0, 0, Math.max(0, options.length - 1)),
            correctOption: asString(item.correctOption) || options[0] || "",
            explanation: asString(item.explanation) || "",
            difficulty: normalizeDifficulty(item.difficulty),
            topic: asString(item.topic) || "General review",
            sourceIds,
          } satisfies ExamSimulationQuestionStateDoc;
        })
        .filter((question): question is ExamSimulationQuestionStateDoc => question !== null)
    : [];

  const allSources = Array.isArray(data.allSources)
    ? data.allSources
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          const id = asString(item.id);
          const type = asString(item.type);
          const label = asString(item.label);
          const snippet = asString(item.snippet);
          if (!id || !label || !snippet || (type !== "chat" && type !== "material")) return null;
          return {
            id,
            type: type as StudySourceType,
            label,
            snippet,
          } satisfies StudySourceDoc;
        })
        .filter((source): source is StudySourceDoc => source !== null)
    : [];

  const remainingQuestionIds = Array.isArray(data.remainingQuestionIds)
    ? data.remainingQuestionIds.map((item) => asString(item)).filter((item) => item.length > 0)
    : [];
  const servedQuestionIds = Array.isArray(data.servedQuestionIds)
    ? data.servedQuestionIds.map((item) => asString(item)).filter((item) => item.length > 0)
    : [];
  const internalResponses = Array.isArray(data.internalResponses)
    ? data.internalResponses
        .map((row) => {
          if (!row || typeof row !== "object") return null;
          const item = row as Record<string, unknown>;
          const questionId = asString(item.questionId);
          const sectionId = asString(item.sectionId);
          const confidence = normalizeExamConfidence(item.confidence);
          if (!questionId || !sectionId || !confidence) return null;
          return {
            questionId,
            questionOrder: clampInteger(asOptionalNumber(item.questionOrder), 0, 0, 100000),
            selectedOptionIndex: clampInteger(asOptionalNumber(item.selectedOptionIndex), 0, 0, 100),
            confidence,
            elapsedSec: clampInteger(asOptionalNumber(item.elapsedSec), 0, 0, 100000),
            answeredAt: asDate(item.answeredAt) || new Date(0),
            difficulty: normalizeDifficulty(item.difficulty),
            topic: asString(item.topic) || "General review",
            sectionId,
            isCorrect: !!item.isCorrect,
          } satisfies ExamSimulationInternalResponseDoc;
        })
        .filter((response): response is ExamSimulationInternalResponseDoc => response !== null)
        .sort((a, b) => a.questionOrder - b.questionOrder)
    : [];

  return {
    id: asString(data.id) || examSimulationId,
    examSimulationId,
    userId: asString(data.userId),
    status: normalizeExamSimulationStatus(data.status),
    currentDifficulty: normalizeDifficulty(data.currentDifficulty),
    questionBank,
    allSources,
    remainingQuestionIds,
    servedQuestionIds,
    currentQuestionId: asString(data.currentQuestionId) || null,
    internalResponses,
    createdAt: asDate(data.createdAt) || new Date(0),
    updatedAt: asDate(data.updatedAt) || new Date(0),
    startedAt: asDate(data.startedAt),
    endsAt: asDate(data.endsAt),
    finishedAt: asDate(data.finishedAt),
  };
}

function applyFlashcardRating(card: StudyFlashcardDoc, rating: FlashcardReviewRating, now: Date): StudyFlashcardDoc {
  let easeFactor = card.easeFactor || 2.5;
  let repetitions = Math.max(0, card.repetitions || 0);
  let intervalDays = Math.max(1, card.intervalDays || 1);

  if (rating === "again") {
    repetitions = 0;
    intervalDays = 1;
    easeFactor = Math.max(1.3, easeFactor - 0.2);
  } else if (rating === "hard") {
    repetitions += 1;
    intervalDays = repetitions <= 1 ? 1 : Math.max(2, Math.round(intervalDays * 1.2));
    easeFactor = Math.max(1.3, easeFactor - 0.15);
  } else if (rating === "good") {
    repetitions += 1;
    intervalDays = repetitions === 1 ? 1 : repetitions === 2 ? 3 : Math.max(2, Math.round(intervalDays * easeFactor));
  } else {
    repetitions += 1;
    intervalDays = repetitions === 1 ? 2 : repetitions === 2 ? 5 : Math.max(3, Math.round(intervalDays * easeFactor * 1.3));
    easeFactor = Math.max(1.3, easeFactor + 0.15);
  }

  const nextReviewAt = new Date(now.getTime() + intervalDays * 24 * 60 * 60 * 1000);
  return {
    ...card,
    easeFactor: Math.round(easeFactor * 100) / 100,
    repetitions,
    intervalDays,
    nextReviewAt,
    lastReviewedAt: now,
  };
}

function parseStoredFlashcards(value: unknown): StudyFlashcardDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyFlashcardDoc[] = [];

  for (const row of value) {
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const front = asString(data.front);
    const back = asString(data.back);
    if (!front || !back) continue;

    out.push({
      id: asString(data.id) || `F${out.length + 1}`,
      front,
      back,
      tags: Array.isArray(data.tags)
        ? data.tags.map((item) => asString(item)).filter((tag) => tag.length > 0).slice(0, 6)
        : [],
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: Array.isArray(data.sourceIds)
        ? data.sourceIds.map((item) => asString(item)).filter((id) => id.length > 0).slice(0, 6)
        : [],
      nextReviewAt: asDate(data.nextReviewAt) || new Date(),
      intervalDays: clampInteger(asOptionalNumber(data.intervalDays), 1, 1, 3650),
      easeFactor: Math.max(1.3, asOptionalNumber(data.easeFactor) ?? 2.5),
      repetitions: Math.max(0, clampInteger(asOptionalNumber(data.repetitions), 0, 0, 1000)),
      lastReviewedAt: asDate(data.lastReviewedAt),
    });
  }

  return out;
}

async function loadMaterialById(materialId: string): Promise<CourseMaterialDoc | null> {
  const ref = db.collection("courseMaterials").doc(materialId);
  const snap = await ref.get();
  if (!snap.exists) return null;
  const data = snap.data() as Record<string, unknown>;

  return {
    id: snap.id,
    userId: typeof data.userId === "string" ? data.userId : "",
    courseId: typeof data.courseId === "string" ? data.courseId : null,
    sessionId: typeof data.sessionId === "string" ? data.sessionId : null,
    chatId: typeof data.chatId === "string" ? data.chatId : null,
    fileName: typeof data.fileName === "string" ? data.fileName : snap.id,
    extension: typeof data.extension === "string" ? data.extension : null,
    mimeType: typeof data.mimeType === "string" ? data.mimeType : null,
    storagePath: typeof data.storagePath === "string" ? data.storagePath : "",
    fileType: typeof data.fileType === "string" ? (data.fileType as MaterialFileType) : undefined,
    status: typeof data.status === "string" ? data.status : undefined,
  };
}

async function buildChunksForMaterial(material: CourseMaterialDoc): Promise<{ fileType: MaterialFileType; chunks: MaterialChunkDoc[] }> {
  const file = storage.bucket().file(material.storagePath);
  const [exists] = await file.exists();
  if (!exists) {
    throw new Error("Uploaded file not found in storage");
  }

  const [buffer] = await file.download();
  if (buffer.byteLength > MAX_SOURCE_FILE_SIZE_BYTES) {
    throw new Error("File is too large to process (max 25MB)");
  }

  const extracted = await extractSegmentsFromMaterial(buffer, material);
  const chunks = buildChunkDocs(material, extracted.fileType, extracted.segments);
  if (chunks.length === 0) {
    throw new Error("Could not extract any indexable content from this file");
  }

  return { fileType: extracted.fileType, chunks };
}

/**
 * POST /focus/start
 * Body: { userId: string, courseId?: string, sessionId?: string, mode?: "study_mode"|"exam_simulation", chatId?: string, examSimulationId?: string }
 */
export const focusStart = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const courseId = asOptionalString(body.courseId);
      const sessionId = asOptionalString(body.sessionId);
      const mode = normalizeFocusSessionMode(body.mode);
      const chatId = asOptionalString(body.chatId);
      const examSimulationId = asOptionalString(body.examSimulationId);
      if (!userId) {
        badRequest(res, "Missing required field: userId");
        return;
      }

      const focusSessionId = crypto.randomUUID();
      await db.collection("focusSessions").doc(focusSessionId).set({
        id: focusSessionId,
        userId,
        source: "webcam",
        status: "active",
        courseId: courseId ?? null,
        sessionId: sessionId ?? null,
        mode,
        chatId: chatId ?? null,
        examSimulationId: examSimulationId ?? null,
        startedAt: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      okJson(res, { ok: true, focusSessionId });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /focus/stop
 * Body: { userId: string, focusSessionId: string }
 */
export const focusStop = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const focusSessionId = asRequiredString(body.focusSessionId);
      if (!userId) {
        badRequest(res, "Missing required field: userId");
        return;
      }
      if (!focusSessionId) {
        badRequest(res, "Missing required field: focusSessionId");
        return;
      }

      const fsRef = db.collection("focusSessions").doc(focusSessionId);
      const fsDoc = await fsRef.get();
      if (!fsDoc.exists) {
        sendErrorResponse(res, 404, "Focus session not found");
        return;
      }
      const fs = fsDoc.data() as { userId?: string } | undefined;
      if (!fs || fs.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      await fsRef.set({ status: "ended", endedAt: new Date(), updatedAt: new Date() }, { merge: true });

      okJson(res, { ok: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /gamificationApplyFocusSession
 * Body: { userId: string, focusSessionId: string, timezone?: string }
 */
export const gamificationApplyFocusSession = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const focusSessionId = asRequiredString(body.focusSessionId);
      const timezone = normalizeTimeZone(asOptionalString(body.timezone));

      if (!userId) {
        badRequest(res, "Missing required field: userId");
        return;
      }
      if (!focusSessionId) {
        badRequest(res, "Missing required field: focusSessionId");
        return;
      }

      const focusSessionRef = db.collection("focusSessions").doc(focusSessionId);
      const focusSummaryRef = db.collection("focusSummaries").doc(focusSessionId);

      const [focusSessionSnap, focusSummarySnap] = await Promise.all([
        focusSessionRef.get(),
        focusSummaryRef.get(),
      ]);

      if (!focusSessionSnap.exists) {
        sendErrorResponse(res, 404, "Focus session not found");
        return;
      }
      if (!focusSummarySnap.exists) {
        sendErrorResponse(res, 404, "Focus summary not found");
        return;
      }

      const focusSession = parseFocusSessionDoc(
        focusSessionSnap.id,
        (focusSessionSnap.data() || {}) as Record<string, unknown>,
      );
      if (!focusSession.userId || focusSession.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }
      if (focusSession.status !== "ended") {
        sendErrorResponse(res, 409, "Focus session is not ended");
        return;
      }

      const focusSummary = parseFocusSummaryDoc((focusSummarySnap.data() || {}) as Record<string, unknown>);
      if (focusSummary.userId && focusSummary.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      const anchorDate =
        focusSession.endedAt ||
        (focusSummary.endTs != null ? new Date(focusSummary.endTs * 1000) : null) ||
        focusSummary.createdAt ||
        new Date();
      const weekInfo = getWeekInfoForDate(anchorDate, timezone);

      const startedAtMs = asTimestampMs(focusSession.startedAt);
      const endedAtMs = asTimestampMs(focusSession.endedAt);
      const measuredDurationMs =
        startedAtMs != null && endedAtMs != null && endedAtMs >= startedAtMs
          ? endedAtMs - startedAtMs
          : null;
      const maxAllowedFocusedMs = Math.max(
        0,
        Math.min(
          MAX_FOCUSED_MINUTES_PER_SESSION * 60 * 1000,
          measuredDurationMs ?? MAX_FOCUSED_MINUTES_PER_SESSION * 60 * 1000,
        ),
      );
      const focusedMs = Math.max(0, Math.min(focusSummary.focusedMs, maxAllowedFocusedMs));
      const focusedMinutes = Math.floor(focusedMs / 60000);
      const baseXp = focusedMinutes;
      const qualityMultiplier = getQualityMultiplier(focusSummary.focusPercent);
      const xpGain = Math.round(baseXp * qualityMultiplier);
      const now = new Date();

      const profileRef = db.collection("gamificationProfiles").doc(userId);
      const dailyRef = db.collection("gamificationDailyStats").doc(`${userId}_${weekInfo.dayKey}`);
      const weeklyRef = db.collection("gamificationWeeklyStats").doc(`${userId}_${weekInfo.weekKey}`);
      const awardRef = db.collection("gamificationSessionAwards").doc(focusSessionId);

      const transactionResult = await db.runTransaction(async (tx) => {
        const awardSnap = await tx.get(awardRef);
        const profileSnap = await tx.get(profileRef);

        if (awardSnap.exists) {
          const existingAward = parseSessionAwardDoc(
            focusSessionId,
            userId,
            (awardSnap.data() || {}) as Record<string, unknown>,
          );
          const existingProfile = parseGamificationProfileDoc(
            userId,
            profileSnap.exists ? (profileSnap.data() as Record<string, unknown>) : null,
            now,
            weekInfo.weekKey,
            weekInfo.weekStartDayKey,
          );

          return {
            alreadyProcessed: true,
            award: existingAward,
            profile: existingProfile,
          };
        }

        const dailySnap = await tx.get(dailyRef);
        const weeklySnap = await tx.get(weeklyRef);

        const profile = parseGamificationProfileDoc(
          userId,
          profileSnap.exists ? (profileSnap.data() as Record<string, unknown>) : null,
          now,
          weekInfo.weekKey,
          weekInfo.weekStartDayKey,
        );

        const dailyData = (dailySnap.data() || {}) as Record<string, unknown>;
        const dailyFocusedMinutesBefore = Math.max(0, clampInteger(asOptionalNumber(dailyData.focusedMinutes), 0, 0, 10000000));
        const dailySessionCountBefore = Math.max(0, clampInteger(asOptionalNumber(dailyData.sessionCount), 0, 0, 10000000));
        const dailyQualifiedBefore = Boolean(dailyData.qualified);

        const dailyFocusedMinutesAfter = dailyFocusedMinutesBefore + focusedMinutes;
        const dailyQualifiedAfter = dailyFocusedMinutesAfter >= DAILY_STREAK_TARGET_MINUTES;
        const newlyQualifiedToday = dailyQualifiedAfter && !dailyQualifiedBefore;
        const dailyQualifiedAtBefore = asDate(dailyData.qualifiedAt);
        const dailyQualifiedAtAfter = dailyQualifiedAfter
          ? (dailyQualifiedAtBefore || now)
          : null;

        const weeklyData = (weeklySnap.data() || {}) as Record<string, unknown>;
        const weeklyTargetMinutes = Math.max(
          1,
          clampInteger(asOptionalNumber(weeklyData.targetMinutes), profile.weeklyGoalTargetMinutes, 1, 1000000),
        );
        const weeklyFocusedMinutesBefore = Math.max(0, clampInteger(asOptionalNumber(weeklyData.focusedMinutes), 0, 0, 10000000));
        const weeklySessionCountBefore = Math.max(0, clampInteger(asOptionalNumber(weeklyData.sessionCount), 0, 0, 10000000));
        const weeklyCompletedBefore = Boolean(weeklyData.completed);
        const weeklyCompletedAtBefore = asDate(weeklyData.completedAt);

        const weeklyFocusedMinutesAfter = weeklyFocusedMinutesBefore + focusedMinutes;
        const weeklyCompletedAfter = weeklyFocusedMinutesAfter >= weeklyTargetMinutes;
        const newlyCompletedWeeklyGoal = weeklyCompletedAfter && !weeklyCompletedBefore;
        const weeklyCompletedAtAfter = weeklyCompletedBefore
          ? (weeklyCompletedAtBefore || now)
          : (weeklyCompletedAfter ? now : null);

        let currentStreakDays = profile.currentStreakDays;
        let longestStreakDays = profile.longestStreakDays;
        let lastQualifiedDayKey = profile.lastQualifiedDayKey;
        if (newlyQualifiedToday) {
          if (lastQualifiedDayKey !== weekInfo.dayKey) {
            if (lastQualifiedDayKey && isNextDayKey(lastQualifiedDayKey, weekInfo.dayKey)) {
              currentStreakDays += 1;
            } else {
              currentStreakDays = 1;
            }
            lastQualifiedDayKey = weekInfo.dayKey;
          }
          longestStreakDays = Math.max(longestStreakDays, currentStreakDays);
        }

        const totalXpAfter = profile.totalXp + xpGain;
        const totalFocusedMinutesAfter = profile.totalFocusedMinutes + focusedMinutes;
        const levelAfter = getLevelFromTotalXp(totalXpAfter);

        const unlockedBadgeSet = new Set<GamificationBadgeId>(profile.unlockedBadges);
        const newBadgeUnlocks: GamificationBadgeId[] = [];
        const unlockBadge = (badgeId: GamificationBadgeId, shouldUnlock: boolean) => {
          if (!shouldUnlock || unlockedBadgeSet.has(badgeId)) return;
          unlockedBadgeSet.add(badgeId);
          newBadgeUnlocks.push(badgeId);
        };

        unlockBadge("first_focus_day", newlyQualifiedToday);
        unlockBadge("streak_3", newlyQualifiedToday && currentStreakDays >= 3);
        unlockBadge("streak_7", newlyQualifiedToday && currentStreakDays >= 7);
        unlockBadge("streak_14", newlyQualifiedToday && currentStreakDays >= 14);
        unlockBadge("streak_30", newlyQualifiedToday && currentStreakDays >= 30);
        unlockBadge("weekly_goal_1", newlyCompletedWeeklyGoal);
        unlockBadge("focus_300m", profile.totalFocusedMinutes < 300 && totalFocusedMinutesAfter >= 300);
        unlockBadge("focus_1000m", profile.totalFocusedMinutes < 1000 && totalFocusedMinutesAfter >= 1000);

        let currentWeekKey = profile.currentWeekKey;
        let currentWeekStartDayKey = profile.currentWeekStartDayKey;
        let currentWeekFocusedMinutes = profile.currentWeekFocusedMinutes;
        let currentWeekCompletedAt = profile.currentWeekCompletedAt;
        const shouldUpdateCurrentWeek =
          !currentWeekStartDayKey || weekInfo.weekStartDayKey >= currentWeekStartDayKey;
        if (shouldUpdateCurrentWeek) {
          currentWeekKey = weekInfo.weekKey;
          currentWeekStartDayKey = weekInfo.weekStartDayKey;
          currentWeekFocusedMinutes = weeklyFocusedMinutesAfter;
          currentWeekCompletedAt = weeklyCompletedAtAfter;
        }

        const updatedProfile: GamificationProfileDoc = {
          ...profile,
          currentStreakDays,
          longestStreakDays,
          lastQualifiedDayKey,
          totalXp: totalXpAfter,
          level: levelAfter,
          totalFocusedMinutes: totalFocusedMinutesAfter,
          weeklyGoalTargetMinutes: weeklyTargetMinutes,
          currentWeekKey,
          currentWeekStartDayKey,
          currentWeekFocusedMinutes,
          currentWeekCompletedAt,
          unlockedBadges: Array.from(unlockedBadgeSet),
          updatedAt: now,
        };

        const awardDoc: GamificationSessionAwardDoc = {
          id: focusSessionId,
          focusSessionId,
          userId,
          dayKey: weekInfo.dayKey,
          weekKey: weekInfo.weekKey,
          weekStartDayKey: weekInfo.weekStartDayKey,
          timezone,
          focusedMinutes,
          baseXp,
          qualityMultiplier: Math.round(qualityMultiplier * 100) / 100,
          xpGain,
          focusPercent: focusSummary.focusPercent,
          badgeUnlocks: newBadgeUnlocks,
          createdAt: now,
        };

        tx.set(profileRef, {
          ...updatedProfile,
          id: userId,
        }, { merge: true });

        tx.set(dailyRef, {
          id: dailyRef.id,
          userId,
          dayKey: weekInfo.dayKey,
          weekKey: weekInfo.weekKey,
          weekStartDayKey: weekInfo.weekStartDayKey,
          timezone,
          focusedMinutes: dailyFocusedMinutesAfter,
          sessionCount: dailySessionCountBefore + 1,
          qualified: dailyQualifiedAfter,
          qualifiedAt: dailyQualifiedAtAfter,
          createdAt: asDate(dailyData.createdAt) || now,
          updatedAt: now,
        }, { merge: true });

        tx.set(weeklyRef, {
          id: weeklyRef.id,
          userId,
          weekKey: weekInfo.weekKey,
          weekStartDayKey: weekInfo.weekStartDayKey,
          timezone,
          targetMinutes: weeklyTargetMinutes,
          focusedMinutes: weeklyFocusedMinutesAfter,
          sessionCount: weeklySessionCountBefore + 1,
          completed: weeklyCompletedAfter,
          completedAt: weeklyCompletedAtAfter,
          createdAt: asDate(weeklyData.createdAt) || now,
          updatedAt: now,
        }, { merge: true });

        tx.set(awardRef, {
          ...awardDoc,
          createdAt: now,
          updatedAt: now,
        }, { merge: false });

        return {
          alreadyProcessed: false,
          award: awardDoc,
          profile: updatedProfile,
        };
      });

      okJson(res, {
        ok: true,
        alreadyProcessed: transactionResult.alreadyProcessed,
        award: toGamificationAwardResponse(transactionResult.award),
        profile: toGamificationProfileResponse(transactionResult.profile),
      });
    } catch (error) {
      sendServerError(res, error);
    }
  },
);

/**
 * POST /studyCoach
 */
export const studyCoach = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const mode = asRequiredString(body.mode) as StudyCoachMode | null;
      const phase = asRequiredString(body.phase);
      const eventType = asRequiredString(body.eventType);
      const sprintIndex = asRequiredNumber(body.sprintIndex);
      const elapsedSec = asRequiredNumber(body.elapsedSec);
      const remainingSec = asRequiredNumber(body.remainingSec);
      const focusPercent = asOptionalNumber(body.focusPercent);
      const distractionCount = asOptionalNumber(body.distractionCount);
      const firstDriftSec = asOptionalNumber(body.firstDriftSec);

      if (!userId) {
        badRequest(res, "Missing required field: userId");
        return;
      }
      if (!mode || (mode !== "nudge" && mode !== "recap")) {
        badRequest(res, "Missing or invalid required field: mode");
        return;
      }
      if (!phase) {
        badRequest(res, "Missing required field: phase");
        return;
      }
      if (!eventType) {
        badRequest(res, "Missing required field: eventType");
        return;
      }
      if (sprintIndex == null) {
        badRequest(res, "Missing required field: sprintIndex");
        return;
      }
      if (elapsedSec == null) {
        badRequest(res, "Missing required field: elapsedSec");
        return;
      }
      if (remainingSec == null) {
        badRequest(res, "Missing required field: remainingSec");
        return;
      }

      const instruction =
        mode === "nudge"
          ? "Write one short real-time coaching nudge. Keep it under 22 words."
          : "Write one short sprint recap. Mention performance and one concrete next goal in under 40 words.";

      const prompt = [
        instruction,
        "Tone: supportive, direct, non-judgmental, no emojis, no markdown.",
        `eventType=${eventType}`,
        `phase=${phase}`,
        `sprintIndex=${sprintIndex}`,
        `elapsedSec=${elapsedSec}`,
        `remainingSec=${remainingSec}`,
        `focusPercent=${focusPercent == null ? "unknown" : focusPercent.toFixed(1)}`,
        `distractionCount=${distractionCount == null ? "unknown" : distractionCount}`,
        `firstDriftSec=${firstDriftSec == null ? "unknown" : firstDriftSec}`,
      ].join("\n");

      const coachResponse: GenerateResponse = await ai.generate({
        model: CHAT_MODEL,
        system:
          "You are an AI study coach for Pomodoro sessions. Be concise, specific, and actionable. Return plain text only.",
        prompt,
        config: {
          temperature: 0.5,
          maxOutputTokens: mode === "nudge" ? 64 : 96,
        },
      });

      const rawMessage = coachResponse.text ?? "";
      const cleaned = rawMessage
        .replace(/\s+/g, " ")
        .replace(/^["']+|["']+$/g, "")
        .trim();

      const fallback =
        mode === "nudge"
          ? "Stay with this sprint. One small focused step right now."
          : "Solid effort this sprint. Keep the next sprint focused and aim to improve your focused minutes.";

      okJson(res, { ok: true, message: cleaned.length > 0 ? cleaned : fallback });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /materialIndex
 * Body: { userId: string, materialId: string }
 */
export const materialIndex = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const materialId = asRequiredString(body.materialId);

    if (!userId || !materialId) {
      badRequest(res, "Missing required fields: userId, materialId");
      return;
    }

    const materialRef = db.collection("courseMaterials").doc(materialId);

    try {
      const material = await loadMaterialById(materialId);
      if (!material) {
        sendErrorResponse(res, 404, "Material not found");
        return;
      }
      if (!material.userId || material.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }
      if (!material.storagePath) {
        sendErrorResponse(res, 400, "Material has no storagePath");
        return;
      }

      await materialRef.set(
        {
          status: "processing",
          errorMessage: null,
          updatedAt: new Date(),
        },
        { merge: true }
      );

      const startedAt = Date.now();
      await deleteMaterialChunks(db, material.id);
      const { fileType, chunks } = await buildChunksForMaterial(material);
      await writeMaterialChunks(db, chunks);

      await materialRef.set(
        {
          fileType,
          extension: getFileExtension(material.fileName),
          status: "indexed",
          chunkCount: chunks.length,
          processingMs: Date.now() - startedAt,
          errorMessage: null,
          updatedAt: new Date(),
        },
        { merge: true }
      );

      okJson(res, { ok: true, materialId, chunkCount: chunks.length, fileType });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown material indexing error";
      await materialRef.set(
        {
          status: "failed",
          errorMessage: message,
          updatedAt: new Date(),
        },
        { merge: true }
      );
      sendErrorResponse(res, 500, message);
    }
  }
);

/**
 * POST /materialDelete
 * Body: { userId: string, materialId: string }
 */
export const materialDelete = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const materialId = asRequiredString(body.materialId);
      if (!userId || !materialId) {
        badRequest(res, "Missing required fields: userId, materialId");
        return;
      }

      const material = await loadMaterialById(materialId);
      if (!material) {
        sendErrorResponse(res, 404, "Material not found");
        return;
      }
      if (material.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      await deleteMaterialChunks(db, material.id);

      if (material.storagePath) {
        try {
          await storage.bucket().file(material.storagePath).delete({ ignoreNotFound: true });
        } catch (error) {
          console.warn("Failed to delete storage object for material", { materialId, error });
        }
      }

      await db.collection("courseMaterials").doc(materialId).delete();
      okJson(res, { ok: true, materialId });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /studyGenerate
 * Body: { userId: string, chatId: string, quizCount?: number, flashcardCount?: number, examCount?: number, model?: string }
 */
export const studyGenerate = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const chatId = asRequiredString(body.chatId);
    const selectedModelName = resolveRequestedChatModelName(body.model);
    const selectedChatModel = getChatModel(selectedModelName);
    const quizCount = clampInteger(asOptionalNumber(body.quizCount), DEFAULT_QUIZ_COUNT, 4, 20);
    const flashcardCount = clampInteger(asOptionalNumber(body.flashcardCount), DEFAULT_FLASHCARD_COUNT, 6, 30);
    const examCount = clampInteger(asOptionalNumber(body.examCount), DEFAULT_EXAM_COUNT, 2, 10);

    if (!userId || !chatId) {
      badRequest(res, "Missing required fields: userId, chatId");
      return;
    }

    const ownership = await loadChatOwnership(chatId);
    if (!ownership) {
      sendErrorResponse(res, 404, "Chat not found");
      return;
    }
    if (ownership.userId !== userId) {
      sendErrorResponse(res, 403, "Not allowed");
      return;
    }

    const now = new Date();
    const studySetRef = db.collection("studySets").doc();

    await studySetRef.set({
      id: studySetRef.id,
      userId,
      chatId,
      courseId: ownership.courseId,
      sessionId: ownership.sessionId,
      status: "generating",
      quizQuestions: [],
      flashcards: [],
      examQuestions: [],
      sources: [],
      model: selectedModelName,
      generationMs: null,
      errorMessage: null,
      createdAt: now,
      updatedAt: now,
    } satisfies StudySetDoc);

    try {
      const recentMessages = await loadRecentChatMessages(chatId, userId, MAX_CHAT_MESSAGES_FOR_STUDY_SET);
      if (recentMessages.length === 0) {
        throw new Error("No chat messages found. Ask at least one question before generating a study set.");
      }

      const retrievalSeed = recentMessages
        .filter((msg) => !msg.isAI)
        .slice(-6)
        .map((msg) => msg.text)
        .join("\n");
      const ragCitations = await retrieveRankedCitations(chatId, userId, retrievalSeed || recentMessages[recentMessages.length - 1]?.text || "study topic");

      const { sources: chatSources, transcript } = buildChatSourcesAndTranscript(recentMessages);
      const materialSources = buildMaterialSources(ragCitations);
      const allSources = [...chatSources, ...materialSources];
      const allowedSourceIds = new Set(allSources.map((source) => source.id));

      const prompt = buildStudyGenerationPrompt({
        quizCount,
        flashcardCount,
        examCount,
        transcript,
        sources: allSources,
      });

      const startedAtMs = Date.now();
      const generation: GenerateResponse = await ai.generate({
        model: selectedChatModel,
        system: "You create source-grounded study materials. Return JSON only.",
        prompt,
        config: {
          temperature: 0.35,
          maxOutputTokens: 2600,
        },
      });

      const rawOutput = generation.text || "";
      if (!rawOutput.trim()) {
        throw new Error("Model returned an empty study set");
      }

      const parsed = parseJsonObject(rawOutput);
      const quizQuestions = parseStudyQuizQuestions(parsed.quiz, quizCount, allowedSourceIds);
      const flashcards = parseStudyFlashcards(parsed.flashcards, flashcardCount, allowedSourceIds, now);
      const examQuestions = parseStudyExamQuestions(parsed.examQuestions, examCount, allowedSourceIds);

      if (quizQuestions.length === 0 || flashcards.length === 0 || examQuestions.length === 0) {
        throw new Error("Generated study set was incomplete. Try again with a richer chat context.");
      }

      await studySetRef.set({
        status: "ready",
        quizQuestions,
        flashcards,
        examQuestions,
        sources: allSources,
        generationMs: Date.now() - startedAtMs,
        errorMessage: null,
        updatedAt: new Date(),
      }, { merge: true });

      okJson(res, {
        ok: true,
        studySetId: studySetRef.id,
        quizCount: quizQuestions.length,
        flashcardCount: flashcards.length,
        examCount: examQuestions.length,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Study set generation failed";
      await studySetRef.set({
        status: "failed",
        errorMessage: message,
        updatedAt: new Date(),
      }, { merge: true });
      sendErrorResponse(res, 500, message);
    }
  }
);

/**
 * POST /examSimulationCreate
 * Body: { userId: string, chatId: string, model?: string }
 */
export const examSimulationCreate = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const chatId = asRequiredString(body.chatId);
    const selectedModelName = resolveRequestedChatModelName(body.model);
    const selectedChatModel = getChatModel(selectedModelName);

    if (!userId || !chatId) {
      badRequest(res, "Missing required fields: userId, chatId");
      return;
    }

    const ownership = await loadChatOwnership(chatId);
    if (!ownership) {
      sendErrorResponse(res, 404, "Chat not found");
      return;
    }
    if (ownership.userId !== userId) {
      sendErrorResponse(res, 403, "Not allowed");
      return;
    }

    const now = new Date();
    const examSimulationRef = db.collection("examSimulations").doc();
    const examSimulationStateRef = db.collection("examSimulationStates").doc(examSimulationRef.id);

    await examSimulationRef.set({
      id: examSimulationRef.id,
      userId,
      chatId,
      courseId: ownership.courseId,
      sessionId: ownership.sessionId,
      status: "generating",
      preset: "standard_mock",
      durationSec: EXAM_SIMULATION_DURATION_SEC,
      servedQuestionCount: EXAM_SIMULATION_SERVED_QUESTION_COUNT,
      questionBankCount: EXAM_SIMULATION_BANK_COUNT,
      model: selectedModelName,
      sections: cloneExamSections(),
      servedQuestions: [],
      responses: [],
      currentQuestionId: null,
      recap: null,
      errorMessage: null,
      createdAt: now,
      updatedAt: now,
      startedAt: null,
      endsAt: null,
      finishedAt: null,
    } satisfies ExamSimulationDoc);

    try {
      const recentMessages = await loadRecentChatMessages(chatId, userId, MAX_CHAT_MESSAGES_FOR_STUDY_SET);
      const retrievalSeed = recentMessages
        .filter((msg) => !msg.isAI)
        .slice(-6)
        .map((msg) => msg.text)
        .join("\n");
      const ragCitations = await retrieveRankedCitations(
        chatId,
        userId,
        retrievalSeed || recentMessages[recentMessages.length - 1]?.text || "exam prep topic",
      );

      if (recentMessages.length < 3 && ragCitations.length === 0) {
        throw createHttpError(400, "Add more chat context or indexed materials before generating a mock exam.");
      }

      const { sources: chatSources, transcript } = buildChatSourcesAndTranscript(recentMessages);
      const materialSources = buildMaterialSources(ragCitations);
      const allSources = [...chatSources, ...materialSources];
      const allowedSourceIds = new Set(allSources.map((source) => source.id));
      if (transcript.length < EXAM_SIMULATION_MIN_SOURCE_CHAR_COUNT && materialSources.length === 0) {
        throw createHttpError(400, "The current chat is still too sparse for a grounded mock exam. Add a few more study messages first.");
      }

      const generation: GenerateResponse = await ai.generate({
        model: selectedChatModel,
        system: "You create source-grounded mock exam question banks. Return JSON only.",
        prompt: buildExamSimulationPrompt({
          transcript,
          sources: allSources,
        }),
        config: {
          temperature: 0.25,
          maxOutputTokens: 3200,
        },
      });

      const rawOutput = generation.text || "";
      if (!rawOutput.trim()) {
        throw new Error("Model returned an empty exam question bank");
      }

      const parsed = parseJsonObject(rawOutput);
      const questionBank = parseExamSimulationQuestionBank(parsed.questions, allowedSourceIds);
      if (!hasExpectedExamDifficultyDistribution(questionBank)) {
        throw new Error("Generated exam bank was incomplete. Try again with richer source context.");
      }

      await Promise.all([
        examSimulationStateRef.set({
          id: examSimulationRef.id,
          examSimulationId: examSimulationRef.id,
          userId,
          status: "ready",
          currentDifficulty: "medium",
          questionBank,
          allSources,
          remainingQuestionIds: questionBank.map((question) => question.id),
          servedQuestionIds: [],
          currentQuestionId: null,
          internalResponses: [],
          createdAt: now,
          updatedAt: now,
          startedAt: null,
          endsAt: null,
          finishedAt: null,
        } satisfies ExamSimulationStateDoc),
        examSimulationRef.set({
          status: "ready",
          updatedAt: new Date(),
          errorMessage: null,
        }, { merge: true }),
      ]);

      okJson(res, {
        ok: true,
        examSimulationId: examSimulationRef.id,
      });
    } catch (error) {
      const statusCode = typeof (error as { statusCode?: unknown })?.statusCode === "number"
        ? (error as { statusCode: number }).statusCode
        : 500;
      const message = error instanceof Error ? error.message : "Exam simulation generation failed";
      await examSimulationRef.set({
        status: "failed",
        errorMessage: message,
        updatedAt: new Date(),
      }, { merge: true });
      sendErrorResponse(res, statusCode, message);
    }
  }
);

/**
 * POST /examSimulationStart
 * Body: { userId: string, examSimulationId: string }
 */
export const examSimulationStart = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const examSimulationId = asRequiredString(body.examSimulationId);
    if (!userId || !examSimulationId) {
      badRequest(res, "Missing required fields: userId, examSimulationId");
      return;
    }

    const examSimulationRef = db.collection("examSimulations").doc(examSimulationId);
    const examSimulationStateRef = db.collection("examSimulationStates").doc(examSimulationId);

    try {
      const result = await db.runTransaction(async (tx) => {
        const [examSimulationSnap, examSimulationStateSnap] = await Promise.all([
          tx.get(examSimulationRef),
          tx.get(examSimulationStateRef),
        ]);

        if (!examSimulationSnap.exists || !examSimulationStateSnap.exists) {
          throw createHttpError(404, "Exam simulation not found");
        }

        const examSimulation = parseExamSimulationDoc(
          examSimulationSnap.id,
          (examSimulationSnap.data() || {}) as Record<string, unknown>,
        );
        const examSimulationState = parseExamSimulationStateDoc(
          examSimulationId,
          (examSimulationStateSnap.data() || {}) as Record<string, unknown>,
        );

        if (examSimulation.userId !== userId || examSimulationState.userId !== userId) {
          throw createHttpError(403, "Not allowed");
        }

        if (examSimulation.status === "ready") {
          const firstQuestion = pickNextExamQuestion(
            examSimulationState.questionBank,
            examSimulationState.remainingQuestionIds,
            "medium",
          );
          if (!firstQuestion) {
            throw createHttpError(409, "Exam simulation is missing a valid starting question");
          }

          const publicQuestion = toPublicExamQuestion(firstQuestion, 0);
          const now = new Date();
          const endsAt = new Date(now.getTime() + examSimulation.durationSec * 1000);
          tx.set(examSimulationRef, {
            status: "in_progress",
            servedQuestions: [publicQuestion],
            currentQuestionId: publicQuestion.id,
            startedAt: now,
            endsAt,
            updatedAt: now,
            errorMessage: null,
          }, { merge: true });
          tx.set(examSimulationStateRef, {
            status: "in_progress",
            currentDifficulty: publicQuestion.difficulty,
            remainingQuestionIds: examSimulationState.remainingQuestionIds.filter((id) => id !== publicQuestion.id),
            servedQuestionIds: [publicQuestion.id],
            currentQuestionId: publicQuestion.id,
            startedAt: now,
            endsAt,
            updatedAt: now,
          }, { merge: true });

          return {
            status: "in_progress" as ExamSimulationStatus,
            currentQuestion: publicQuestion,
            startedAt: now.toISOString(),
            endsAt: endsAt.toISOString(),
          };
        }

        if (examSimulation.status === "in_progress") {
          const currentQuestion = examSimulation.servedQuestions.find(
            (question) => question.id === examSimulation.currentQuestionId,
          ) || null;
          return {
            status: examSimulation.status,
            currentQuestion,
            startedAt: examSimulation.startedAt ? examSimulation.startedAt.toISOString() : null,
            endsAt: examSimulation.endsAt ? examSimulation.endsAt.toISOString() : null,
          };
        }

        throw createHttpError(409, "This exam simulation cannot be started again");
      });

      okJson(res, {
        ok: true,
        examSimulationId,
        ...result,
      });
    } catch (error) {
      if (typeof (error as { statusCode?: unknown })?.statusCode === "number") {
        sendErrorResponse(res, (error as { statusCode: number }).statusCode, error instanceof Error ? error.message : "Request failed");
        return;
      }
      sendServerError(res, error);
    }
  }
);

/**
 * POST /examSimulationAnswer
 * Body: { userId: string, examSimulationId: string, questionId: string, selectedOptionIndex: number, confidence: "low"|"medium"|"high", elapsedSec: number }
 */
export const examSimulationAnswer = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const examSimulationId = asRequiredString(body.examSimulationId);
    const questionId = asRequiredString(body.questionId);
    const selectedOptionIndex = asRequiredNumber(body.selectedOptionIndex);
    const confidence = normalizeExamConfidence(body.confidence);
    const elapsedSec = clampInteger(asOptionalNumber(body.elapsedSec), 0, 0, EXAM_SIMULATION_DURATION_SEC);

    if (!userId || !examSimulationId || !questionId || selectedOptionIndex == null || !confidence) {
      badRequest(res, "Missing required fields: userId, examSimulationId, questionId, selectedOptionIndex, confidence");
      return;
    }

    const examSimulationRef = db.collection("examSimulations").doc(examSimulationId);
    const examSimulationStateRef = db.collection("examSimulationStates").doc(examSimulationId);

    try {
      const result = await db.runTransaction(async (tx) => {
        const [examSimulationSnap, examSimulationStateSnap] = await Promise.all([
          tx.get(examSimulationRef),
          tx.get(examSimulationStateRef),
        ]);

        if (!examSimulationSnap.exists || !examSimulationStateSnap.exists) {
          throw createHttpError(404, "Exam simulation not found");
        }

        const examSimulation = parseExamSimulationDoc(
          examSimulationSnap.id,
          (examSimulationSnap.data() || {}) as Record<string, unknown>,
        );
        const examSimulationState = parseExamSimulationStateDoc(
          examSimulationId,
          (examSimulationStateSnap.data() || {}) as Record<string, unknown>,
        );

        if (examSimulation.userId !== userId || examSimulationState.userId !== userId) {
          throw createHttpError(403, "Not allowed");
        }
        if (examSimulation.status !== "in_progress") {
          throw createHttpError(409, "Exam simulation is not running");
        }
        if (examSimulation.currentQuestionId !== questionId || examSimulationState.currentQuestionId !== questionId) {
          throw createHttpError(409, "This is not the active question");
        }
        if (examSimulation.endsAt && Date.now() > examSimulation.endsAt.getTime()) {
          throw createHttpError(409, "Exam time has already expired");
        }
        if (examSimulationState.internalResponses.some((response) => response.questionId === questionId)) {
          throw createHttpError(409, "This question has already been answered");
        }

        const privateQuestion = examSimulationState.questionBank.find((question) => question.id === questionId);
        if (!privateQuestion) {
          throw createHttpError(404, "Question not found in exam state");
        }
        if (!Number.isInteger(selectedOptionIndex) || selectedOptionIndex < 0 || selectedOptionIndex >= privateQuestion.options.length) {
          throw createHttpError(400, "Selected option is out of range");
        }

        const questionOrder = examSimulationState.internalResponses.length;
        const currentPublicQuestion = examSimulation.servedQuestions.find((question) => question.id === questionId);
        const sectionId = currentPublicQuestion?.sectionId || getExamSectionForOrder(questionOrder).id;
        const answeredAt = new Date();
        const responseBase = {
          questionId,
          questionOrder,
          selectedOptionIndex,
          confidence,
          elapsedSec,
          answeredAt,
          difficulty: privateQuestion.difficulty,
          topic: privateQuestion.topic,
          sectionId,
        } satisfies ExamSimulationResponseDoc;
        const isCorrect = selectedOptionIndex === privateQuestion.correctOptionIndex;
        const internalResponse: ExamSimulationInternalResponseDoc = {
          ...responseBase,
          isCorrect,
        };
        const publicResponses = [...examSimulation.responses, responseBase];
        const internalResponses = [...examSimulationState.internalResponses, internalResponse];
        const answeredCount = internalResponses.length;
        const reachedQuestionLimit = answeredCount >= examSimulation.servedQuestionCount;
        const targetDifficulty = determineNextExamTargetDifficulty(privateQuestion.difficulty, isCorrect, confidence);

        let nextPublicQuestion: ExamSimulationQuestionDoc | null = null;
        let nextCurrentDifficulty = targetDifficulty;
        let remainingQuestionIds = [...examSimulationState.remainingQuestionIds];
        let servedQuestionIds = [...examSimulationState.servedQuestionIds];
        let servedQuestions = [...examSimulation.servedQuestions];

        if (!reachedQuestionLimit) {
          const nextQuestion = pickNextExamQuestion(
            examSimulationState.questionBank,
            remainingQuestionIds,
            targetDifficulty,
          );
          if (nextQuestion) {
            nextPublicQuestion = toPublicExamQuestion(nextQuestion, answeredCount);
            nextCurrentDifficulty = nextQuestion.difficulty;
            remainingQuestionIds = remainingQuestionIds.filter((id) => id !== nextQuestion.id);
            servedQuestionIds = [...servedQuestionIds, nextQuestion.id];
            servedQuestions = [...servedQuestions, nextPublicQuestion];
          }
        }

        tx.set(examSimulationRef, {
          responses: publicResponses,
          servedQuestions,
          currentQuestionId: nextPublicQuestion?.id ?? null,
          updatedAt: answeredAt,
        }, { merge: true });
        tx.set(examSimulationStateRef, {
          internalResponses,
          servedQuestionIds,
          remainingQuestionIds,
          currentDifficulty: nextCurrentDifficulty,
          currentQuestionId: nextPublicQuestion?.id ?? null,
          updatedAt: answeredAt,
        }, { merge: true });

        return {
          status: "in_progress" as ExamSimulationStatus,
          done: nextPublicQuestion == null || reachedQuestionLimit,
          currentQuestion: nextPublicQuestion,
        };
      });

      okJson(res, {
        ok: true,
        examSimulationId,
        ...result,
      });
    } catch (error) {
      if (typeof (error as { statusCode?: unknown })?.statusCode === "number") {
        sendErrorResponse(res, (error as { statusCode: number }).statusCode, error instanceof Error ? error.message : "Request failed");
        return;
      }
      sendServerError(res, error);
    }
  }
);

/**
 * POST /examSimulationFinish
 * Body: { userId: string, examSimulationId: string, completionReason: "submitted"|"time_up"|"abandoned", focusSessionId?: string }
 */
export const examSimulationFinish = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const examSimulationId = asRequiredString(body.examSimulationId);
    const completionReasonRaw = asRequiredString(body.completionReason);
    const focusSessionId = asOptionalString(body.focusSessionId);

    if (!userId || !examSimulationId || !completionReasonRaw) {
      badRequest(res, "Missing required fields: userId, examSimulationId, completionReason");
      return;
    }

    const completionReason = completionReasonRaw === "time_up" || completionReasonRaw === "abandoned" || completionReasonRaw === "submitted"
      ? completionReasonRaw
      : null;
    if (!completionReason) {
      badRequest(res, "completionReason must be one of: submitted, time_up, abandoned");
      return;
    }

    const examSimulationRef = db.collection("examSimulations").doc(examSimulationId);
    const examSimulationStateRef = db.collection("examSimulationStates").doc(examSimulationId);

    try {
      const [examSimulationSnap, examSimulationStateSnap] = await Promise.all([
        examSimulationRef.get(),
        examSimulationStateRef.get(),
      ]);
      if (!examSimulationSnap.exists || !examSimulationStateSnap.exists) {
        sendErrorResponse(res, 404, "Exam simulation not found");
        return;
      }

      const examSimulation = parseExamSimulationDoc(
        examSimulationSnap.id,
        (examSimulationSnap.data() || {}) as Record<string, unknown>,
      );
      const examSimulationState = parseExamSimulationStateDoc(
        examSimulationId,
        (examSimulationStateSnap.data() || {}) as Record<string, unknown>,
      );
      if (examSimulation.userId !== userId || examSimulationState.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      if (
        (examSimulation.status === "completed" ||
          examSimulation.status === "timed_out" ||
          examSimulation.status === "abandoned") &&
        examSimulation.recap
      ) {
        okJson(res, {
          ok: true,
          examSimulationId,
          status: examSimulation.status,
          recap: examSimulation.recap,
        });
        return;
      }

      const finalStatus: ExamSimulationStatus =
        completionReason === "time_up"
          ? "timed_out"
          : completionReason === "abandoned"
            ? "abandoned"
            : "completed";
      const recap = await buildExamSimulationRecap({
        userId,
        modelName: examSimulation.model,
        sections: examSimulation.sections,
        allSources: examSimulationState.allSources,
        responses: examSimulationState.internalResponses,
        focusSessionId: focusSessionId ?? null,
      });
      const now = new Date();

      await db.runTransaction(async (tx) => {
        const currentExamSnap = await tx.get(examSimulationRef);
        const currentStateSnap = await tx.get(examSimulationStateRef);
        if (!currentExamSnap.exists || !currentStateSnap.exists) {
          throw createHttpError(404, "Exam simulation not found");
        }
        const currentExam = parseExamSimulationDoc(
          currentExamSnap.id,
          (currentExamSnap.data() || {}) as Record<string, unknown>,
        );
        const currentState = parseExamSimulationStateDoc(
          examSimulationId,
          (currentStateSnap.data() || {}) as Record<string, unknown>,
        );
        if (currentExam.userId !== userId || currentState.userId !== userId) {
          throw createHttpError(403, "Not allowed");
        }

        tx.set(examSimulationRef, {
          status: finalStatus,
          currentQuestionId: null,
          recap,
          finishedAt: now,
          updatedAt: now,
          errorMessage: null,
        }, { merge: true });
        tx.set(examSimulationStateRef, {
          status: finalStatus,
          currentQuestionId: null,
          finishedAt: now,
          updatedAt: now,
        }, { merge: true });
      });

      okJson(res, {
        ok: true,
        examSimulationId,
        status: finalStatus,
        recap,
      });
    } catch (error) {
      if (typeof (error as { statusCode?: unknown })?.statusCode === "number") {
        sendErrorResponse(res, (error as { statusCode: number }).statusCode, error instanceof Error ? error.message : "Request failed");
        return;
      }
      sendServerError(res, error);
    }
  }
);

/**
 * POST /flashcardReview
 * Body: { userId: string, studySetId: string, cardId: string, rating: "again"|"hard"|"good"|"easy" }
 */
export const flashcardReview = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const studySetId = asRequiredString(body.studySetId);
    const cardId = asRequiredString(body.cardId);
    const ratingRaw = asRequiredString(body.rating);
    const validRatings: FlashcardReviewRating[] = ["again", "hard", "good", "easy"];
    const rating = validRatings.includes((ratingRaw || "") as FlashcardReviewRating)
      ? (ratingRaw as FlashcardReviewRating)
      : null;

    if (!userId || !studySetId || !cardId || !rating) {
      badRequest(res, "Missing required fields: userId, studySetId, cardId, rating");
      return;
    }

    const studySetRef = db.collection("studySets").doc(studySetId);
    const studySetSnap = await studySetRef.get();
    if (!studySetSnap.exists) {
      sendErrorResponse(res, 404, "Study set not found");
      return;
    }
    const data = studySetSnap.data() as Record<string, unknown>;
    if (asString(data.userId) !== userId) {
      sendErrorResponse(res, 403, "Not allowed");
      return;
    }

    const flashcards = parseStoredFlashcards(data.flashcards);
    const cardIndex = flashcards.findIndex((card) => card.id === cardId);
    if (cardIndex < 0) {
      sendErrorResponse(res, 404, "Flashcard not found");
      return;
    }

    const now = new Date();
    const updatedCard = applyFlashcardRating(flashcards[cardIndex], rating, now);
    flashcards[cardIndex] = updatedCard;

    await studySetRef.set({
      flashcards,
      updatedAt: now,
    }, { merge: true });

    await db.collection("flashcardReviews").add({
      userId,
      studySetId,
      chatId: asString(data.chatId) || null,
      cardId,
      rating,
      intervalDays: updatedCard.intervalDays,
      easeFactor: updatedCard.easeFactor,
      repetitions: updatedCard.repetitions,
      reviewedAt: now,
      createdAt: now,
    });

    okJson(res, {
      ok: true,
      cardId: updatedCard.id,
      nextReviewAt: updatedCard.nextReviewAt.toISOString(),
      intervalDays: updatedCard.intervalDays,
      easeFactor: updatedCard.easeFactor,
      repetitions: updatedCard.repetitions,
    });
  }
);

export const chat = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const body = req.body || {};
      const sessionId = asRequiredString(body.sessionId);
      const message = asRequiredString(body.message);
      const userId = asRequiredString(body.userId);
      const selectedModelName = resolveRequestedChatModelName(body.model);
      const selectedChatModel = getChatModel(selectedModelName);
      if (!sessionId || !message || !userId) {
        sendErrorResponse(res, 400, "Missing required fields: sessionId, message, userId");
        return;
      }

      let session = await ai.loadSession(sessionId, { store: sessionStore });
      if (!session) {
        await sessionStore.save(sessionId, {
          id: sessionId,
          state: {},
          threads: { [MAIN_THREAD]: [] },
        });
        session = await ai.loadSession(sessionId, { store: sessionStore });
      }
      if (!session) {
        sendErrorResponse(res, 500, "Failed to initialize session");
        return;
      }

      const sessionData = await sessionStore.get(session.id);
      const chatHistory = fromThreadMessages(sessionData?.threads?.[MAIN_THREAD]);
      setSSEHeaders(res);

      const publicChatRef = db.collection("chats").doc(sessionId);
      const publicChatDoc = await publicChatRef.get();
      const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";

      const ragCitations = await retrieveRankedCitations(sessionId, userId, message);
      const ragPrompt = buildRagUserPrompt(message, ragCitations);

      let fullText = "";
      const modelMessages = toGenkitMessages(chatHistory, ragPrompt);
      const generationOptions: GenerateOptions = {
        model: selectedChatModel,
        system:
          ragCitations.length > 0
            ? `${SYSTEM_INSTRUCTION}\n\nUploaded-material snippets may be included as optional references. Use them when relevant and cite snippet-backed claims with [C#]. You can still answer with general knowledge for topics outside the snippets.`
            : `${SYSTEM_INSTRUCTION}\n\nAnswer normally using general knowledge. If the user asks about uploaded files and no relevant snippets are available, say that and suggest uploading/indexing the material.`,
        messages: modelMessages,
        config: { temperature: 0.5 },
        onChunk: (chunk: GenerateResponseChunk) => {
          const chunkText = chunk.text;
          if (chunkText) {
            fullText += chunkText;
            res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
          }
        },
      };
      const finalResponse: GenerateResponse = await ai.generate(generationOptions);
      const finalText = finalResponse.text;
      if (typeof finalText === "string" && finalText.length >= fullText.length) {
        fullText = finalText;
      }

      let newSessionName = null;
      if (isNewChat) {
        try {
          const titleOptions: GenerateOptions = {
            model: selectedChatModel,
            system: "You are a helpful assistant that generates short, concise titles for chat sessions.",
            prompt: `Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`,
            config: { temperature: 0.2, maxOutputTokens: 24 },
          };
          const titleResult: GenerateResponse = await ai.generate(titleOptions);
          const rawTitle = titleResult.text;
          const title = rawTitle.trim().replace(/^['"]+|['"]+$/g, "").trim();
          if (title) {
            await publicChatRef.update({ name: title });
            newSessionName = title;
          }
        } catch (error) {
          console.error("Error generating title:", error);
        }
      }

      const updatedHistory: ChatHistoryMessage[] = [
        ...chatHistory,
        { role: "user", content: message },
        { role: "model", content: fullText },
      ];

      await sessionStore.save(session.id, {
        id: session.id,
        state: sessionData?.state ?? {},
        threads: { [MAIN_THREAD]: toThreadMessages(updatedHistory) },
      });

      const finalCitations = selectFinalCitations(fullText, ragCitations);
      const clientCitations = finalCitations.map((citation) => {
        const { contextText, ...publicCitation } = citation;
        void contextText;
        return publicCitation;
      });
      res.write(
        `data: ${JSON.stringify({
          text: "",
          done: true,
          model: selectedModelName,
          sessionId: session.id,
          fullText,
          newSessionName,
          citations: clientCitations,
        })}\n\n`
      );
      res.end();
    } catch (error) {
      sendServerError(res, error);
    }
  }
);
