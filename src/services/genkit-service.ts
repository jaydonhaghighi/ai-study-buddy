/**
 * Genkit AI Service
 * 
 * This service communicates with Cloud Functions that use Genkit
 * for persistent chat sessions with Gemini.
 */

export interface AIResponse {
  text: string;
  model: string;
  sessionId: string;
}

interface ChatRequest {
  sessionId: string;
  message: string;
  userId: string;
}

interface CreateSessionRequest {
  userId: string;
  sessionName?: string;
}

const getFunctionsUrl = () => {
  const projectId = import.meta.env.VITE_FIREBASE_PROJECT_ID;
  if (!projectId) {
    throw new Error('Firebase project ID is not configured');
  }
  return `https://us-central1-${projectId}.cloudfunctions.net`;
};

export async function getAIResponse(
  userMessage: string,
  sessionId: string,
  userId: string,
  onChunk?: (text: string) => void
): Promise<AIResponse> {
  const functionsUrl = getFunctionsUrl();

  try {
    const response = await fetch(`${functionsUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sessionId,
        message: userMessage,
        userId,
      } as ChatRequest),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    if (response.headers.get('content-type')?.includes('text/event-stream')) {
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let buffer = '';

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.done) {
                return {
                  text: data.fullText || fullText,
                  model: data.model || 'gemini-2.0-flash-exp',
                  sessionId: data.sessionId || sessionId,
                };
              }
              if (data.text) {
                fullText += data.text;
                if (onChunk) {
                  onChunk(data.text);
                }
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }

      return {
        text: fullText,
        model: 'gemini-2.0-flash-exp',
        sessionId: sessionId,
      };
    } else {
      const data = await response.json();
      return {
        text: data.text,
        model: data.model || 'gemini-2.5-flash',
        sessionId: data.sessionId || sessionId,
      };
    }
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to get AI response');
  }
}

export async function createGenkitSession(
  userId: string,
  sessionName?: string
): Promise<{ sessionId: string; name: string }> {
  const functionsUrl = getFunctionsUrl();

  try {
    const response = await fetch(`${functionsUrl}/createSession`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId,
        sessionName,
      } as CreateSessionRequest),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to create session');
  }
}

export async function getSessionHistory(sessionId: string): Promise<{
  history: Array<{ role: string; content: string }>;
  state: any;
}> {
  const functionsUrl = getFunctionsUrl();

  try {
    const response = await fetch(`${functionsUrl}/getSessionHistory?sessionId=${encodeURIComponent(sessionId)}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to get session history');
  }
}

