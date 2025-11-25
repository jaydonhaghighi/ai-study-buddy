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

interface CreateChatRequest {
  userId: string;
  courseId: string;
  sessionId: string;
  chatName?: string;
}

const getFunctionsUrl = () => {
  const projectId = import.meta.env.VITE_FIREBASE_PROJECT_ID;
  if (!projectId) {
    throw new Error('Firebase project ID is not configured');
  }
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

function handleError(error: unknown, defaultMessage: string): never {
  if (error instanceof Error) {
    throw error;
  }
  throw new Error(defaultMessage);
}

export async function getAIResponse(
  userMessage: string,
  sessionId: string,
  userId: string,
  onChunk?: (text: string) => void
): Promise<AIResponse> {
  try {
    const response = await fetchFunction('/chat', {
      method: 'POST',
      body: JSON.stringify({ sessionId, message: userMessage, userId } as ChatRequest),
    });

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
                if (!data.model) {
                  throw new Error('Model name not provided in response');
                }
                return {
                  text: data.fullText || fullText,
                  model: data.model,
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

      throw new Error('Stream ended without model information');
    } else {
      const data = await response.json();
      if (!data.model) {
        throw new Error('Model name not provided in response');
      }
      return {
        text: data.text,
        model: data.model,
        sessionId: data.sessionId || sessionId,
      };
    }
  } catch (error) {
    handleError(error, 'Failed to get AI response');
  }
}

export async function createGenkitChat(
  userId: string,
  courseId: string,
  sessionId: string,
  chatName?: string
): Promise<{ chatId: string; name: string }> {
  try {
    const response = await fetchFunction('/createChat', {
      method: 'POST',
      body: JSON.stringify({ userId, courseId, sessionId, chatName } as CreateChatRequest),
    });
    return await response.json();
  } catch (error) {
    handleError(error, 'Failed to create chat');
  }
}

export async function getSessionHistory(sessionId: string): Promise<{
  history: Array<{ role: string; content: string }>;
  state: any;
}> {
  try {
    const response = await fetchFunction(`/getSessionHistory?sessionId=${encodeURIComponent(sessionId)}`, {
      method: 'GET',
    });
    return await response.json();
  } catch (error) {
    handleError(error, 'Failed to get session history');
  }
}

