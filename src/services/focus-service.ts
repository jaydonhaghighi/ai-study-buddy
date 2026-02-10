/**
 * Focus tracking service (web client)
 * Calls Firebase Cloud Functions (HTTP onRequest exports).
 */

const getFunctionsUrl = () => {
  const projectId = import.meta.env.VITE_FIREBASE_PROJECT_ID;
  if (!projectId) throw new Error('Firebase project ID is not configured');
  return `https://us-central1-${projectId}.cloudfunctions.net`;
};

async function fetchFunction(endpoint: string, options: RequestInit = {}) {
  const functionsUrl = getFunctionsUrl();
  const response = await fetch(`${functionsUrl}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
  }

  return response;
}

export async function startFocusSession(params: {
  userId: string;
  courseId?: string;
  sessionId?: string;
  source?: 'webcam';
}): Promise<{ ok: boolean; focusSessionId: string }> {
  const res = await fetchFunction('/focusStart', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return await res.json();
}

export async function stopFocusSession(params: {
  userId: string;
  focusSessionId: string;
}): Promise<{ ok: boolean }> {
  const res = await fetchFunction('/focusStop', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return await res.json();
}


