function getFunctionsUrl(): string {
  const projectId = import.meta.env.VITE_FIREBASE_PROJECT_ID;
  if (!projectId) {
    throw new Error('Firebase project ID is not configured');
  }
  return `https://us-central1-${projectId}.cloudfunctions.net`;
}

export async function fetchFunctionsEndpoint(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const functionsUrl = getFunctionsUrl();
  const headers = new Headers(options.headers);
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${functionsUrl}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let message = `HTTP error! status: ${response.status}`;
    try {
      const errorData = await response.json();
      if (typeof errorData?.error === 'string' && errorData.error.trim().length > 0) {
        message = errorData.error;
      } else if (typeof errorData?.message === 'string' && errorData.message.trim().length > 0) {
        message = errorData.message;
      }
    } catch {
      const text = await response.text().catch(() => '');
      if (text.trim().length > 0) {
        message = text;
      } else {
        message = 'Unknown error';
      }
    }
    throw new Error(message);
  }

  return response;
}
