export interface DataItem {
  id: string;
  text: string;
  userId: string;
  createdAt: Date | null;
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
