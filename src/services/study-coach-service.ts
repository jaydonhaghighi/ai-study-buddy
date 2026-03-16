import { fetchFunctionsEndpoint } from './functions-http';

export type StudyCoachMode = 'nudge' | 'recap';

export type StudyCoachEventType =
  | 'sprint_start'
  | 'distracted_sustained_20s'
  | 'back_in_focus'
  | 'last_minute'
  | 'sprint_end_recap';

export type StudyCoachRequest = {
  userId: string;
  mode: StudyCoachMode;
  phase: string;
  eventType: StudyCoachEventType;
  sprintIndex: number;
  elapsedSec: number;
  remainingSec: number;
  focusPercent?: number;
  distractionCount?: number;
  firstDriftSec?: number;
};

export type StudyCoachResponse = {
  ok: boolean;
  message: string;
};

export async function getStudyCoachMessage(payload: StudyCoachRequest): Promise<StudyCoachResponse> {
  const response = await fetchFunctionsEndpoint('/studyCoach', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  return response.json();
}
