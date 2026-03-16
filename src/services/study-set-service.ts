import { fetchFunctionsEndpoint } from './functions-http';
import type { FlashcardReviewRating } from '../types';

export type StudySetPreset = 'quick' | 'standard' | 'exam';

type GenerateStudySetParams = {
  userId: string;
  chatId: string;
  preset?: StudySetPreset;
};

type SubmitFlashcardReviewParams = {
  userId: string;
  studySetId: string;
  cardId: string;
  rating: FlashcardReviewRating;
};

function presetCounts(preset: StudySetPreset): { quizCount: number; flashcardCount: number; examCount: number } {
  if (preset === 'quick') {
    return { quizCount: 6, flashcardCount: 10, examCount: 2 };
  }
  if (preset === 'exam') {
    return { quizCount: 12, flashcardCount: 16, examCount: 5 };
  }
  return { quizCount: 10, flashcardCount: 14, examCount: 4 };
}

export async function generateStudySet({
  userId,
  chatId,
  preset = 'standard',
}: GenerateStudySetParams): Promise<{ studySetId: string }> {
  const counts = presetCounts(preset);
  const response = await fetchFunctionsEndpoint('/studyGenerate', {
    method: 'POST',
    body: JSON.stringify({
      userId,
      chatId,
      quizCount: counts.quizCount,
      flashcardCount: counts.flashcardCount,
      examCount: counts.examCount,
    }),
  });

  const payload = (await response.json()) as { studySetId?: string };
  if (!payload.studySetId || typeof payload.studySetId !== 'string') {
    throw new Error('Study set ID missing in response');
  }
  return { studySetId: payload.studySetId };
}

export async function submitFlashcardReview({
  userId,
  studySetId,
  cardId,
  rating,
}: SubmitFlashcardReviewParams): Promise<void> {
  await fetchFunctionsEndpoint('/flashcardReview', {
    method: 'POST',
    body: JSON.stringify({
      userId,
      studySetId,
      cardId,
      rating,
    }),
  });
}
