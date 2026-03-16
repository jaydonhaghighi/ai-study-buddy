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

