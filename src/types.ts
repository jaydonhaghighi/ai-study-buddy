export interface DataItem {
  id: string;
  text: string;
  userId: string;
  createdAt: Date | null;
}

export type FocusAlertSettings = {
  nudgeDelayMinutes: number;
  soundEnabled: boolean;
  volume: number;
  updatedAt: Date | null;
};

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

export type ExamSimulationStatus =
  | 'generating'
  | 'ready'
  | 'in_progress'
  | 'completed'
  | 'timed_out'
  | 'abandoned'
  | 'failed';

export type ExamConfidence = 'low' | 'medium' | 'high';
export type ExamCompletionReason = 'submitted' | 'time_up' | 'abandoned';

export type ExamSimulationSection = {
  id: string;
  label: string;
  questionTargetCount: number;
  targetDurationSec: number;
  cumulativeTargetSec: number;
};

export type ExamSimulationQuestion = {
  id: string;
  prompt: string;
  options: string[];
  difficulty: StudyDifficulty;
  topic: string;
  sourceIds: string[];
  sectionId: string;
  orderIndex: number;
};

export type ExamSimulationResponse = {
  questionId: string;
  questionOrder: number;
  selectedOptionIndex: number;
  confidence: ExamConfidence;
  elapsedSec: number;
  answeredAt: Date | null;
  difficulty: StudyDifficulty;
  topic: string;
  sectionId: string;
};

export type ExamSimulationWeakTopic = {
  topic: string;
  accuracyPercent: number;
  questionCount: number;
  correctCount: number;
};

export type ExamSimulationDifficultyBreakdown = {
  difficulty: StudyDifficulty;
  accuracyPercent: number;
  questionCount: number;
  correctCount: number;
};

export type ExamSimulationSectionResult = {
  sectionId: string;
  label: string;
  targetDurationSec: number;
  actualElapsedSec: number;
  overrunSec: number;
  answeredCount: number;
};

export type ExamSimulationFocusInsight = {
  focusSessionId: string;
  focusPercent: number;
  distractions: number;
  firstDriftOffsetSec: number | null;
};

export type ExamSimulationRecap = {
  answeredCount: number;
  totalQuestionCount: number;
  correctCount: number;
  scorePercent: number;
  completionPercent: number;
  overconfidenceMisses: number;
  weakTopics: ExamSimulationWeakTopic[];
  accuracyByDifficulty: ExamSimulationDifficultyBreakdown[];
  sectionResults: ExamSimulationSectionResult[];
  timeLossMoments: string[];
  recoveryPlan: string[];
  weakTopicSummary: string;
  focusInsight: ExamSimulationFocusInsight | null;
};

export type ExamSimulation = {
  id: string;
  userId: string;
  chatId: string;
  courseId: string | null;
  sessionId: string | null;
  status: ExamSimulationStatus;
  preset: 'standard_mock';
  durationSec: number;
  servedQuestionCount: number;
  questionBankCount: number;
  model: string;
  sections: ExamSimulationSection[];
  servedQuestions: ExamSimulationQuestion[];
  responses: ExamSimulationResponse[];
  currentQuestionId: string | null;
  recap: ExamSimulationRecap | null;
  errorMessage: string | null;
  createdAt: Date | null;
  updatedAt: Date | null;
  startedAt: Date | null;
  endsAt: Date | null;
  finishedAt: Date | null;
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
