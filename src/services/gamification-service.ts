import type { GamificationApplyResponse } from '../types';
import { fetchFunctionsEndpoint } from './functions-http';

export async function applyGamificationForFocusSession(params: {
  userId: string;
  focusSessionId: string;
  timezone?: string;
}): Promise<GamificationApplyResponse> {
  const timezone = params.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

  const res = await fetchFunctionsEndpoint('/gamificationApplyFocusSession', {
    method: 'POST',
    body: JSON.stringify({
      userId: params.userId,
      focusSessionId: params.focusSessionId,
      timezone,
    }),
  });

  return res.json();
}
