/**
 * Genkit AI Service
 *
 * This service communicates with Cloud Functions that use Genkit
 * for persistent chat sessions.
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
      let finalModel: string | null = null;
      let finalSessionId: string | null = null;

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const processLine = (line: string) => {
        if (!line.startsWith('data: ')) return null;
        try {
          const data = JSON.parse(line.slice(6));
          if (data?.sessionId && typeof data.sessionId === 'string') {
            finalSessionId = data.sessionId;
          }
          if (data?.model && typeof data.model === 'string') {
            finalModel = data.model;
          }
          if (data?.done) {
            return {
              text: (typeof data.fullText === 'string' && data.fullText.trim().length > 0) ? data.fullText : fullText,
              model: finalModel ?? 'unknown',
              sessionId: finalSessionId ?? sessionId,
            } satisfies AIResponse;
          }
          if (data?.text && typeof data.text === 'string') {
            fullText += data.text;
            onChunk?.(data.text);
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
        return null;
      };

      while (true) {
        // eslint-disable-next-line no-await-in-loop
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const maybe = processLine(line);
          if (maybe) return maybe;
        }
      }

      // Stream ended. Try to parse any remaining buffered line(s) that didn't end with '\n'.
      if (buffer.trim().length > 0) {
        for (const line of buffer.split('\n')) {
          const maybe = processLine(line);
          if (maybe) return maybe;
        }
      }

      // If we received text but no final metadata, return what we have instead of throwing.
      if (fullText.trim().length > 0) {
        return { text: fullText, model: finalModel ?? 'unknown', sessionId: finalSessionId ?? sessionId };
      }

      throw new Error('Stream ended without any content');
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

