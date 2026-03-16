export interface DataItem {
  id: string;
  text: string;
  userId: string;
  createdAt: Date | null;
}

// Exam Simulation Types
export interface ExamQuestion {
  id: string;
  topic: string;
  text: string;
  options: string[];
  correctAnswer: number;
  initialDifficulty: 'easy' | 'medium' | 'hard';
  explanation: string;
}

export interface ExamSection {
  id: string;
  title: string;
  description: string;
  timePerQuestionMs: number;
  totalQuestionsTarget: number;
  questions: ExamQuestion[];
}

export interface ExamConfiguration {
  id: string;
  title: string;
  description: string;
  totalTimeMs: number;
  sections: ExamSection[];
  adaptiveDifficulty: boolean;
}

export interface AnsweredQuestion {
  questionId: string;
  topic: string;
  selectedAnswerIndex: number;
  isCorrect: boolean;
  difficulty: 'easy' | 'medium' | 'hard';
  timeSpentMs: number;
  confidence: 'low' | 'medium' | 'high';
}

export interface ExamSession {
  id: string;
  userId: string;
  examId: string;
  examTitle: string;
  startedAt: Date;
  completedAt?: Date;
  answeredQuestions: AnsweredQuestion[];
  currentSectionId: string;
  currentQuestionIndex: number;
  totalTimeUsedMs: number;
  status: 'in-progress' | 'completed' | 'abandoned';
}

export interface PerformanceMetrics {
  totalQuestions: number;
  correctAnswers: number;
  accuracyPercent: number;
  averageTimePerQuestion: number;
  timeWastedMs: number;
  confidenceAlignment: number;
}

export interface TopicPerformance {
  topic: string;
  questionsAttempted: number;
  correctAnswers: number;
  accuracyPercent: number;
  averageTimeMs: number;
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface ExamReport {
  sessionId: string;
  examTitle: string;
  completedAt: Date;
  totalDurationMs: number;
  performanceMetrics: PerformanceMetrics;
  topicPerformance: TopicPerformance[];
  weakTopics: string[];
  strongTopics: string[];
  timeLossAreas: { moment: string; extraTimeMs: number }[];
  recoveryPlan: string[];
}

export type Citation = {
  id: string;
  materialId: string;
  fileName: string;
  fileType: 'pdf' | 'docx' | 'spreadsheet' | 'slides' | 'txt' | 'image';
  locationType: 'page' | 'sheet' | 'slide' | 'line' | 'image';
  locationLabel: string;
  snippet: string;
  score?: number;
};

export type CourseMaterialStatus = 'uploading' | 'uploaded' | 'processing' | 'indexed' | 'failed';

export type CourseMaterial = {
  id: string;
  userId: string;
  courseId: string | null;
  sessionId: string | null;
  chatId: string | null;
  fileName: string;
  extension: string;
  mimeType: string;
  sizeBytes: number;
  storagePath: string;
  fileType: 'pdf' | 'docx' | 'spreadsheet' | 'slides' | 'txt' | 'image' | '';
  status: CourseMaterialStatus;
  chunkCount?: number;
  errorMessage?: string | null;
  processingMs?: number;
  createdAt: Date | null;
  updatedAt: Date | null;
};

export type StudySetStatus = 'generating' | 'ready' | 'failed';
export type StudyDifficulty = 'easy' | 'medium' | 'hard';
export type QuizQuestionType = 'mcq' | 'short';
export type FlashcardReviewRating = 'again' | 'hard' | 'good' | 'easy';

export type StudySource = {
  id: string;
  type: 'chat' | 'material';
  label: string;
  snippet: string;
};

export type QuizQuestion = {
  id: string;
  questionType: QuizQuestionType;
  prompt: string;
  options: string[];
  correctAnswer: string;
  correctOptionIndex: number | null;
  explanation: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

export type Flashcard = {
  id: string;
  front: string;
  back: string;
  tags: string[];
  difficulty: StudyDifficulty;
  sourceIds: string[];
  nextReviewAt: Date | null;
  intervalDays: number;
  easeFactor: number;
  repetitions: number;
  lastReviewedAt: Date | null;
};

export type ExamQuestion = {
  id: string;
  prompt: string;
  rubric: string[];
  modelAnswer: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

export type StudySet = {
  id: string;
  userId: string;
  chatId: string;
  courseId: string | null;
  sessionId: string | null;
  status: StudySetStatus;
  quizQuestions: QuizQuestion[];
  flashcards: Flashcard[];
  examQuestions: ExamQuestion[];
  sources: StudySource[];
  model: string;
  generationMs: number | null;
  errorMessage: string | null;
  createdAt: Date | null;
  updatedAt: Date | null;
};

export type GamificationBadgeId =
  | 'first_focus_day'
  | 'streak_3'
  | 'streak_7'
  | 'streak_14'
  | 'streak_30'
  | 'weekly_goal_1'
  | 'focus_300m'
  | 'focus_1000m';

export type GamificationProfile = {
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
};

export type GamificationDailyStat = {
  id: string;
  userId: string;
  dayKey: string;
  weekKey: string;
  weekStartDayKey: string;
  timezone: string;
  focusedMinutes: number;
  sessionCount: number;
  qualified: boolean;
  qualifiedAt: Date | null;
  createdAt: Date | null;
  updatedAt: Date | null;
};

export type GamificationWeeklyStat = {
  id: string;
  userId: string;
  weekKey: string;
  weekStartDayKey: string;
  timezone: string;
  targetMinutes: number;
  focusedMinutes: number;
  sessionCount: number;
  completed: boolean;
  completedAt: Date | null;
  createdAt: Date | null;
  updatedAt: Date | null;
};

export type GamificationSessionAward = {
  focusSessionId: string;
  dayKey: string;
  weekKey: string;
  focusedMinutes: number;
  baseXp: number;
  qualityMultiplier: number;
  xpGain: number;
  focusPercent: number;
  badgeUnlocks: GamificationBadgeId[];
};

export type GamificationApplyResponse = {
  ok: boolean;
  alreadyProcessed: boolean;
  award: GamificationSessionAward;
  profile: GamificationProfile;
};
