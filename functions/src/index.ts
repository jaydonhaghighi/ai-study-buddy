import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import crypto from "crypto";
import { genkit, SessionStore, SessionData } from "genkit/beta";
import { openAI } from "@genkit-ai/compat-oai/openai";

initializeApp();
const db = getFirestore();

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
      const data = sessionDoc.data();
      const sessionData: any = {
        id: sessionId,
        state: (data?.state || {}) as S,
      };
      if (data?.history) {
        sessionData.history = data.history;
        console.log("Loaded session with history:", {
          sessionId,
          messageCount: Array.isArray(data.history) ? data.history.length : 0,
        });
      } else if (data?.threads) {
        const threads = data.threads;
        const threadKeys = Object.keys(threads);
        const firstThread = threadKeys.length > 0 ? threads[threadKeys[0]] : [];
        sessionData.history = Array.isArray(firstThread) ? firstThread : [];
        console.log("Migrated threads to history format:", {
          sessionId,
          messageCount: sessionData.history.length,
        });
      } else {
        sessionData.history = [];
        console.log("Loaded session without history:", sessionId);
      }
      return sessionData as SessionData<S>;
    } catch (error) {
      console.error("Error loading session:", error);
      return undefined;
    }
  }

  async save(sessionId: string, data: SessionData<S>): Promise<void> {
    try {
      const saveData: any = {
        id: sessionId,
        state: data.state || {},
        updatedAt: new Date(),
      };
      const dataAny = data as any;
      if (dataAny.history) {
        saveData.history = dataAny.history;
      }
      console.log("Saving session with history:", {
        sessionId,
        messageCount: Array.isArray(dataAny.history) ? dataAny.history.length : 0,
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
      const fs = fsDoc.data() as any;
      if (fs.userId !== userId) {
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

type GenkitMessage = {
  role: "user" | "model";
  content: Array<{ text: string }>;
};

function convertHistoryToMessages(history: any[]): ChatHistoryMessage[] {
  const out: ChatHistoryMessage[] = [];
  for (const h of history) {
    if (!h || typeof h.content !== "string") continue;
    const content = h.content.trim();
    if (!content) continue;
    const role: ChatHistoryMessage["role"] = h.role === "user" ? "user" : "model";
    out.push({ role, content });
  }
  return out;
}

function toGenkitMessages(history: ChatHistoryMessage[], userMessage: string): GenkitMessage[] {
  const messages: GenkitMessage[] = history.map((h) => ({
    role: h.role,
    content: [{ text: h.content }],
  }));
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

async function loadSessionOrError(sessionId: string, res: any) {
  const session = await ai.loadSession(sessionId, { store: sessionStore });
  if (!session) {
    sendErrorResponse(res, 404, "Session not found. The session may have been deleted or never created.");
    return null;
  }
  return session;
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
        await sessionStore.save(sessionId, { id: sessionId, state: {}, history: [] } as any);
        session = await ai.loadSession(sessionId, { store: sessionStore });
      }
      // If it still fails, treat as server error (store issues, etc).
      if (!session) {
        sendErrorResponse(res, 500, "Failed to initialize session");
        return;
      }

      const sessionDoc = await db.collection("genkit_sessions").doc(session.id).get();
      const sessionData = sessionDoc.data() as any;
      const history = sessionData?.history || [];

      const chatHistory = convertHistoryToMessages(history);
      setSSEHeaders(res);

      // Check if we need to generate a title (look in 'chats' collection now)
      const publicChatRef = db.collection("chats").doc(sessionId);
      const publicChatDoc = await publicChatRef.get();
      // Fallback to 'sessions' for backward compatibility if needed, but focusing on new structure
      const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";

      let fullText = "";
      const modelMessages = toGenkitMessages(chatHistory, message);
      const finalResponse = await ai.generate({
        model: openAI.model(MODEL_NAME),
        system: SYSTEM_INSTRUCTION,
        messages: modelMessages,
        config: { temperature: 0.7 },
        onChunk: (chunk: any) => {
          const chunkText =
            typeof (chunk as any).text === "string"
              ? (chunk as any).text
              : typeof (chunk as any).text === "function"
                ? (chunk as any).text()
                : "";
          if (chunkText) {
            fullText += chunkText;
            res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
          }
        },
      } as any);
      const finalText =
        typeof (finalResponse as any).text === "string"
          ? (finalResponse as any).text
          : typeof (finalResponse as any).text === "function"
            ? (finalResponse as any).text()
            : "";
      if (typeof finalText === "string" && finalText.length >= fullText.length) {
        fullText = finalText;
      }

      // Generate title if needed
      let newSessionName = null;
      if (isNewChat) {
        try {
          const titleResult = await ai.generate({
            model: openAI.model(MODEL_NAME),
            system: "You are a helpful assistant that generates short, concise titles for chat sessions.",
            prompt: `Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`,
            config: { temperature: 0.2, maxOutputTokens: 24 },
          } as any);
          const rawTitle =
            typeof (titleResult as any).text === "string"
              ? (titleResult as any).text
              : typeof (titleResult as any).text === "function"
                ? (titleResult as any).text()
                : "";
          const title = rawTitle.trim().replace(/^['"]+|['"]+$/g, "").trim();
          if (title) {
            await publicChatRef.update({ name: title });
            newSessionName = title;
          }
        } catch (error) {
          console.error("Error generating title:", error);
        }
      }

      const updatedHistory = [
        ...chatHistory,
        { role: "user", content: message },
        { role: "model", content: fullText },
      ];

      await sessionStore.save(session.id, {
        id: session.id,
        state: sessionData?.state || {},
        history: updatedHistory,
      } as any);

      res.write(`data: ${JSON.stringify({ text: '', done: true, model: MODEL_NAME, sessionId: session.id, fullText, newSessionName })}\n\n`);
      res.end();
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

export const createChat = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const { userId, chatName, courseId, sessionId } = req.body;

      if (!userId || !courseId || !sessionId) {
        sendErrorResponse(res, 400, "Missing required fields: userId, courseId, sessionId");
        return;
      }

      const session = ai.createSession({ store: sessionStore });
      const name = chatName || "New Chat";
      
      await Promise.all([
        sessionStore.save(session.id, { id: session.id, state: {} }),
        db.collection("chats").doc(session.id).set({
          id: session.id,
          name,
          userId,
          courseId,
          sessionId,
          createdAt: new Date(),
          lastMessageAt: new Date(),
        })
      ]);

      res.json({ chatId: session.id, name });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

export const getSessionHistory = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "GET") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const { sessionId } = req.query;

      if (!sessionId || typeof sessionId !== "string") {
        sendErrorResponse(res, 400, "Missing required query parameter: sessionId");
        return;
      }

      const session = await loadSessionOrError(sessionId, res);
      if (!session) return;

      const sessionData = await sessionStore.get(sessionId);
      res.json({ state: sessionData?.state || {} });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

