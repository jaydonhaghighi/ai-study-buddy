import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import crypto from "crypto";
import { genkit, SessionStore, SessionData } from "genkit/beta";
import { openAI } from "@genkit-ai/compat-oai/openai";
import type {
  GenerateOptions,
  GenerateResponse,
  GenerateResponseChunk,
  MessageData,
} from "genkit";

initializeApp();
const db = getFirestore();
const MAIN_THREAD = "main";

const MODEL_NAME = process.env.OPENAI_MODEL || "gpt-4o-mini";

const ai = genkit({
  plugins: [openAI({
    apiKey: process.env.OPENAI_API_KEY || "",
  })],
  model: openAI.model(MODEL_NAME),
});

interface StudyBuddyState {
  userName?: string;
  preferences?: {
    learningStyle?: string;
    subject?: string;
  };
}

class FirestoreSessionStore<S = any> implements SessionStore<S> {
  async get(sessionId: string): Promise<SessionData<S> | undefined> {
    try {
      const sessionDoc = await db.collection("genkit_sessions").doc(sessionId).get();
      if (!sessionDoc.exists) {
        return undefined;
      }
      const data = sessionDoc.data() as {
        state?: S;
        threads?: Record<string, MessageData[]>;
        history?: Array<{ role?: unknown; content?: unknown }>;
      };
      const sessionData: SessionData<S> = {
        id: sessionId,
        state: (data?.state || {}) as S,
      };

      if (data?.threads && typeof data.threads === "object") {
        sessionData.threads = data.threads;
      }

      // Back-compat: older sessions stored flat "history".
      if (!sessionData.threads) {
        const legacyThread: MessageData[] = Array.isArray(data?.history)
          ? data.history
              .filter((m): m is { role: unknown; content: unknown } => !!m)
              .map((m) => {
                const role: "user" | "model" = m.role === "user" ? "user" : "model";
                return {
                  role,
                  content: [{ text: typeof m.content === "string" ? m.content : "" }],
                };
              })
              .filter((m) => m.content[0].text.trim().length > 0)
          : [];
        sessionData.threads = { [MAIN_THREAD]: legacyThread };
      }

      console.log("Loaded session with history:", {
        sessionId,
        messageCount: sessionData.threads?.[MAIN_THREAD]?.length ?? 0,
      });

      return sessionData as SessionData<S>;
    } catch (error) {
      console.error("Error loading session:", error);
      return undefined;
    }
  }

  async save(sessionId: string, data: SessionData<S>): Promise<void> {
    try {
      const saveData: SessionData<S> & { updatedAt: Date } = {
        id: sessionId,
        state: data.state ?? ({} as S),
        threads: data.threads || { [MAIN_THREAD]: [] },
        updatedAt: new Date(),
      };
      console.log("Saving session with history:", {
        sessionId,
        messageCount: saveData.threads?.[MAIN_THREAD]?.length ?? 0,
      });
      await db.collection("genkit_sessions").doc(sessionId).set(saveData, { merge: true });
      console.log("Session saved successfully with history");
    } catch (error) {
      console.error("Error saving session:", error);
      throw error;
    }
  }
}

const sessionStore = new FirestoreSessionStore<StudyBuddyState>();

const SYSTEM_INSTRUCTION = `You are an AI Study Buddy - a knowledgeable, patient, and encouraging learning companion designed to help students succeed academically.

Your core principles:
- Be friendly, approachable, and supportive in all interactions
- Break down complex concepts into clear, digestible explanations
- Use examples and analogies to make learning more relatable
- Encourage questions and create a safe learning environment
- Adapt your teaching style to the student's level and needs
- Provide step-by-step guidance when explaining processes or solving problems
- Acknowledge effort and progress to build confidence
- Be concise but thorough - avoid overwhelming with too much information at once

When answering questions:
- Start with a clear, direct answer
- Then provide context and deeper explanation if needed
- Use numbered lists or bullet points for multi-step processes
- Include relevant examples or real-world applications
- Ask follow-up questions to check understanding when appropriate

Remember: Your goal is to help students learn effectively, not just provide answers. Foster critical thinking and independent learning skills.`;

const FUNCTION_CONFIG = {
  cors: true,
  region: "us-central1" as const,
  secrets: ["OPENAI_API_KEY"],
};

function okJson(res: any, body: any) {
  res.status(200).json(body);
}

function badRequest(res: any, message: string) {
  sendErrorResponse(res, 400, message);
}

/**
 * POST /focus/start
 * Body: { userId: string, courseId?: string, sessionId?: string }
 */
export const focusStart = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { userId, courseId, sessionId } = req.body || {};
      if (!userId || typeof userId !== "string") {
        badRequest(res, "Missing required field: userId");
        return;
      }

      const focusSessionId = crypto.randomUUID();
      await db.collection("focusSessions").doc(focusSessionId).set({
        id: focusSessionId,
        userId,
        source: "webcam",
        status: "active",
        courseId: (typeof courseId === "string" ? courseId : null),
        sessionId: (typeof sessionId === "string" ? sessionId : null),
        startedAt: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      okJson(res, { ok: true, focusSessionId });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /focus/stop
 * Body: { userId: string, focusSessionId: string }
 */
export const focusStop = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { userId, focusSessionId } = req.body || {};
      if (!userId || typeof userId !== "string") {
        badRequest(res, "Missing required field: userId");
        return;
      }
      if (!focusSessionId || typeof focusSessionId !== "string") {
        badRequest(res, "Missing required field: focusSessionId");
        return;
      }

      const fsRef = db.collection("focusSessions").doc(focusSessionId);
      const fsDoc = await fsRef.get();
      if (!fsDoc.exists) {
        sendErrorResponse(res, 404, "Focus session not found");
        return;
      }
      const fs = fsDoc.data() as { userId?: string } | undefined;
      if (!fs || fs.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      await fsRef.set({ status: "ended", endedAt: new Date(), updatedAt: new Date() }, { merge: true });

      okJson(res, { ok: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

type ChatHistoryMessage = {
  role: "user" | "model";
  content: string;
};

function fromThreadMessages(messages: MessageData[] | undefined): ChatHistoryMessage[] {
  const out: ChatHistoryMessage[] = [];
  for (const message of messages || []) {
    if (message.role !== "user" && message.role !== "model") continue;
    const content = message.content
      .map((part) => (typeof part.text === "string" ? part.text : ""))
      .join("")
      .trim();
    if (!content) continue;
    out.push({ role: message.role, content });
  }
  return out;
}

function toThreadMessages(history: ChatHistoryMessage[]): MessageData[] {
  return history.map((h) => ({
    role: h.role,
    content: [{ text: h.content }],
  }));
}

function toGenkitMessages(history: ChatHistoryMessage[], userMessage: string): MessageData[] {
  const messages = toThreadMessages(history);
  messages.push({
    role: "user",
    content: [{ text: userMessage }],
  });
  return messages;
}

function setSSEHeaders(res: any) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
}

function sendErrorResponse(res: any, status: number, message: string) {
  res.status(status).json({ error: message });
}

function sendServerError(res: any, error: unknown) {
  console.error("Error:", error);
  res.status(500).json({
    error: error instanceof Error ? error.message : "Internal server error",
  });
}

export const chat = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const { sessionId, message, userId } = req.body;

      if (!sessionId || !message || !userId) {
        sendErrorResponse(res, 400, "Missing required fields: sessionId, message, userId");
        return;
      }

      // Lazily initialize Genkit session state on first message.
      // This removes the need for a separate "create chat" round trip and
      // makes chat creation feel instant in the UI.
      let session = await ai.loadSession(sessionId, { store: sessionStore });
      if (!session) {
        await sessionStore.save(sessionId, {
          id: sessionId,
          state: {},
          threads: { [MAIN_THREAD]: [] },
        });
        session = await ai.loadSession(sessionId, { store: sessionStore });
      }
      // If it still fails, treat as server error (store issues, etc).
      if (!session) {
        sendErrorResponse(res, 500, "Failed to initialize session");
        return;
      }

      const sessionData = await sessionStore.get(session.id);
      const chatHistory = fromThreadMessages(sessionData?.threads?.[MAIN_THREAD]);
      setSSEHeaders(res);

      // Check if we need to generate a title (look in 'chats' collection now)
      const publicChatRef = db.collection("chats").doc(sessionId);
      const publicChatDoc = await publicChatRef.get();
      // Fallback to 'sessions' for backward compatibility if needed, but focusing on new structure
      const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";

      let fullText = "";
      const modelMessages = toGenkitMessages(chatHistory, message);
      const generationOptions: GenerateOptions = {
        model: openAI.model(MODEL_NAME),
        system: SYSTEM_INSTRUCTION,
        messages: modelMessages,
        config: { temperature: 0.7 },
        onChunk: (chunk: GenerateResponseChunk) => {
          const chunkText = chunk.text;
          if (chunkText) {
            fullText += chunkText;
            res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
          }
        },
      };
      const finalResponse: GenerateResponse = await ai.generate(generationOptions);
      const finalText = finalResponse.text;
      if (typeof finalText === "string" && finalText.length >= fullText.length) {
        fullText = finalText;
      }

      // Generate title if needed
      let newSessionName = null;
      if (isNewChat) {
        try {
          const titleOptions: GenerateOptions = {
            model: openAI.model(MODEL_NAME),
            system: "You are a helpful assistant that generates short, concise titles for chat sessions.",
            prompt: `Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`,
            config: { temperature: 0.2, maxOutputTokens: 24 },
          };
          const titleResult: GenerateResponse = await ai.generate(titleOptions);
          const rawTitle = titleResult.text;
          const title = rawTitle.trim().replace(/^['"]+|['"]+$/g, "").trim();
          if (title) {
            await publicChatRef.update({ name: title });
            newSessionName = title;
          }
        } catch (error) {
          console.error("Error generating title:", error);
        }
      }

      const updatedHistory: ChatHistoryMessage[] = [
        ...chatHistory,
        { role: "user", content: message },
        { role: "model", content: fullText },
      ];

      await sessionStore.save(session.id, {
        id: session.id,
        state: sessionData?.state || {},
        threads: { [MAIN_THREAD]: toThreadMessages(updatedHistory) },
      });

      res.write(`data: ${JSON.stringify({ text: '', done: true, model: MODEL_NAME, sessionId: session.id, fullText, newSessionName })}\n\n`);
      res.end();
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

