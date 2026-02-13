/**
 * Focus tracking service (web client)
 * Calls Firebase Cloud Functions (HTTP onRequest exports).
 */
import { fetchFunctionsEndpoint } from './functions-http';

export async function startFocusSession(params: {
  userId: string;
  courseId?: string;
  sessionId?: string;
}): Promise<{ ok: boolean; focusSessionId: string }> {
  const res = await fetchFunctionsEndpoint('/focusStart', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function stopFocusSession(params: {
  userId: string;
  focusSessionId: string;
}): Promise<{ ok: boolean }> {
  const res = await fetchFunctionsEndpoint('/focusStop', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return res.json();
}
