import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import crypto from "crypto";
import { genkit, SessionStore, SessionData } from "genkit/beta";
import { googleAI } from "@genkit-ai/googleai";
import { GoogleGenerativeAI } from "@google/generative-ai";

initializeApp();
const db = getFirestore();

const MODEL_NAME = "gemini-2.5-flash";

const ai = genkit({
  plugins: [googleAI({
    apiKey: process.env.GOOGLE_GENAI_API_KEY || "",
  })],
  model: googleAI.model(MODEL_NAME),
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
  secrets: ["GOOGLE_GENAI_API_KEY"],
};

function sha256Base64Url(input: string) {
  const hash = crypto.createHash("sha256").update(input, "utf8").digest("base64");
  return hash.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function newDeviceToken() {
  // 32 bytes -> ~43 chars base64url
  const raw = crypto.randomBytes(32).toString("base64");
  return raw.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function getBearerToken(req: any): string | null {
  const header = req.get?.("Authorization") || req.headers?.authorization;
  if (!header || typeof header !== "string") return null;
  const m = header.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : null;
}

async function findDeviceByClaimCode(claimCode: string) {
  const snap = await db.collection("devices").where("claimCode", "==", claimCode).limit(1).get();
  if (snap.empty) return null;
  return snap.docs[0];
}

async function findDeviceByToken(deviceToken: string) {
  const tokenHash = sha256Base64Url(deviceToken);
  const snap = await db.collection("devices").where("deviceTokenHash", "==", tokenHash).limit(1).get();
  if (snap.empty) return null;
  return snap.docs[0];
}

function okJson(res: any, body: any) {
  res.status(200).json(body);
}

function badRequest(res: any, message: string) {
  sendErrorResponse(res, 400, message);
}

// ------------------------
// Device + Focus endpoints
// ------------------------

/**
 * POST /device/register
 * Body: { claimCode: string, deviceId?: string }
 */
export const deviceRegister = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { claimCode, deviceId } = req.body || {};
      if (!claimCode || typeof claimCode !== "string") {
        badRequest(res, "Missing required field: claimCode");
        return;
      }
      const id = (deviceId && typeof deviceId === "string") ? deviceId : crypto.randomUUID();
      const ref = db.collection("devices").doc(id);

      await ref.set(
        {
          id,
          claimCode,
          status: "awaiting_claim",
          createdAt: new Date(),
          updatedAt: new Date(),
        },
        { merge: true }
      );

      okJson(res, { ok: true, deviceId: id });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /device/claim
 * Body: { claimCode: string, userId: string }
 * Creates a device token (returned via /device/pairingStatus for the Pi).
 */
export const deviceClaim = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { claimCode, userId } = req.body || {};
      if (!claimCode || typeof claimCode !== "string") {
        badRequest(res, "Missing required field: claimCode");
        return;
      }
      if (!userId || typeof userId !== "string") {
        badRequest(res, "Missing required field: userId");
        return;
      }

      const deviceDoc = await findDeviceByClaimCode(claimCode);
      if (!deviceDoc) {
        sendErrorResponse(res, 404, "Device not found for claim code");
        return;
      }

      const token = newDeviceToken();
      const tokenHash = sha256Base64Url(token);

      await deviceDoc.ref.set(
        {
          status: "paired",
          pairedUserId: userId,
          pairedAt: new Date(),
          deviceTokenHash: tokenHash,
          deviceTokenOnce: token, // will be cleared after Pi fetches it
          tokenDelivered: false,
          updatedAt: new Date(),
        },
        { merge: true }
      );

      okJson(res, { ok: true, deviceId: deviceDoc.id, paired: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * GET /device/pairingStatus?claimCode=...
 * Returns: { paired: boolean, deviceToken?: string, deviceId?: string }
 */
export const devicePairingStatus = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "GET") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const claimCode = req.query.claimCode;
      if (!claimCode || typeof claimCode !== "string") {
        badRequest(res, "Missing required query parameter: claimCode");
        return;
      }

      const deviceDoc = await findDeviceByClaimCode(claimCode);
      if (!deviceDoc) {
        okJson(res, { paired: false });
        return;
      }

      const data = deviceDoc.data() as any;
      const paired = data?.status === "paired" && !!data?.pairedUserId;

      // One-time token delivery pattern
      if (paired && data?.deviceTokenOnce && data?.tokenDelivered === false) {
        const token = data.deviceTokenOnce as string;
        await deviceDoc.ref.set(
          {
            tokenDelivered: true,
            deviceTokenOnce: null,
            updatedAt: new Date(),
          },
          { merge: true }
        );
        okJson(res, { paired: true, deviceToken: token, deviceId: deviceDoc.id });
        return;
      }

      okJson(res, { paired, deviceId: deviceDoc.id });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * GET /device/currentFocusSession
 * Header: Authorization: Bearer <deviceToken>
 * Returns: { focusSessionId: string | null, courseId?: string, sessionId?: string }
 */
export const deviceCurrentFocusSession = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "GET") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const deviceToken = getBearerToken(req);
      if (!deviceToken) {
        sendErrorResponse(res, 401, "Missing device token");
        return;
      }

      const deviceDoc = await findDeviceByToken(deviceToken);
      if (!deviceDoc) {
        sendErrorResponse(res, 401, "Invalid device token");
        return;
      }

      const device = deviceDoc.data() as any;
      const focusSessionId = device?.activeFocusSessionId || null;
      if (!focusSessionId) {
        okJson(res, { focusSessionId: null });
        return;
      }

      const fsDoc = await db.collection("focusSessions").doc(String(focusSessionId)).get();
      const fsData = fsDoc.exists ? (fsDoc.data() as any) : {};

      okJson(res, {
        focusSessionId: String(focusSessionId),
        courseId: fsData?.courseId || null,
        sessionId: fsData?.sessionId || null,
      });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /device/sessionSummary
 * Header: Authorization: Bearer <deviceToken>
 * Body: focus summary payload (see pi-agent)
 */
export const deviceSessionSummary = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const deviceToken = getBearerToken(req);
      if (!deviceToken) {
        sendErrorResponse(res, 401, "Missing device token");
        return;
      }
      const deviceDoc = await findDeviceByToken(deviceToken);
      if (!deviceDoc) {
        sendErrorResponse(res, 401, "Invalid device token");
        return;
      }

      const body = req.body || {};
      const focusSessionId = body.focusSessionId;
      if (!focusSessionId || typeof focusSessionId !== "string") {
        badRequest(res, "Missing required field: focusSessionId");
        return;
      }

      const focusSessionDoc = await db.collection("focusSessions").doc(focusSessionId).get();
      const focusSession = focusSessionDoc.exists ? (focusSessionDoc.data() as any) : null;

      const device = deviceDoc.data() as any;
      const userId = focusSession?.userId || device?.pairedUserId;
      if (!userId) {
        sendErrorResponse(res, 400, "Cannot resolve userId for this device/session");
        return;
      }

      // Write summary keyed by focusSessionId for idempotency
      await db.collection("focusSummaries").doc(focusSessionId).set(
        {
          ...body,
          userId,
          deviceId: deviceDoc.id,
          createdAt: new Date(),
        },
        { merge: true }
      );

      // Mark focus session ended
      await db.collection("focusSessions").doc(focusSessionId).set(
        {
          status: "ended",
          endedAt: new Date(),
          updatedAt: new Date(),
        },
        { merge: true }
      );

      // If device currently points at this session, clear it
      const active = device?.activeFocusSessionId;
      if (active && String(active) === focusSessionId) {
        await deviceDoc.ref.set({ activeFocusSessionId: null, updatedAt: new Date() }, { merge: true });
      }

      okJson(res, { ok: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /focus/start
 * Body: { userId: string, deviceId?: string, courseId?: string, sessionId?: string, source?: "pi" | "webcam" }
 */
export const focusStart = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { userId, deviceId, courseId, sessionId, source } = req.body || {};
      if (!userId || typeof userId !== "string") {
        badRequest(res, "Missing required field: userId");
        return;
      }

      const usingDevice = typeof deviceId === "string" && deviceId.length > 0;
      let deviceRef: any = null;
      if (usingDevice) {
        deviceRef = db.collection("devices").doc(deviceId);
        const deviceDoc = await deviceRef.get();
        if (!deviceDoc.exists) {
          sendErrorResponse(res, 404, "Device not found");
          return;
        }
      }

      const focusSessionId = crypto.randomUUID();
      await db.collection("focusSessions").doc(focusSessionId).set({
        id: focusSessionId,
        userId,
        deviceId: usingDevice ? deviceId : null,
        source: source === "webcam" || !usingDevice ? "webcam" : "pi",
        status: "active",
        courseId: (typeof courseId === "string" ? courseId : null),
        sessionId: (typeof sessionId === "string" ? sessionId : null),
        startedAt: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      if (deviceRef) {
        await deviceRef.set(
          { activeFocusSessionId: focusSessionId, updatedAt: new Date() },
          { merge: true }
        );
      }

      okJson(res, { ok: true, focusSessionId });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /focus/stop
 * Body: { userId: string, focusSessionId: string, deviceId?: string }
 */
export const focusStop = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const { userId, focusSessionId, deviceId } = req.body || {};
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

      const devId = (typeof deviceId === "string" ? deviceId : fs.deviceId);
      if (devId) {
        const devRef = db.collection("devices").doc(String(devId));
        const devDoc = await devRef.get();
        if (devDoc.exists) {
          const dev = devDoc.data() as any;
          if (dev?.activeFocusSessionId && String(dev.activeFocusSessionId) === focusSessionId) {
            await devRef.set({ activeFocusSessionId: null, updatedAt: new Date() }, { merge: true });
          }
        }
      }

      okJson(res, { ok: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

function getGenerativeModel(systemInstruction?: string) {
  const apiKey = process.env.GOOGLE_GENAI_API_KEY || "";
  const genAI = new GoogleGenerativeAI(apiKey);
  return genAI.getGenerativeModel({ 
    model: MODEL_NAME,
    systemInstruction: systemInstruction,
  });
}

function getStudyBuddyModel() {
  return getGenerativeModel(SYSTEM_INSTRUCTION);
}

function convertHistoryToChatFormat(history: any[]): Array<{ role: string; parts: Array<{ text: string }> }> {
  return history
    .filter((h: any) => h && h.role && h.content && typeof h.content === "string")
    .map((h: any) => ({
      role: h.role === "user" ? "user" : "model",
      parts: [{ text: h.content }],
    }))
    .filter((h: any) => h.parts[0].text.trim().length > 0);
}

function ensureHistoryStartsWithUser(chatHistory: Array<{ role: string; parts: Array<{ text: string }> }>) {
  while (chatHistory.length > 0 && chatHistory[0].role !== "user") {
    chatHistory.shift();
  }
  return chatHistory;
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

      const session = await loadSessionOrError(sessionId, res);
      if (!session) return;

      const model = getStudyBuddyModel();
      const sessionDoc = await db.collection("genkit_sessions").doc(session.id).get();
      const sessionData = sessionDoc.data() as any;
      const history = sessionData?.history || [];

      let chatHistory = convertHistoryToChatFormat(history);
      chatHistory = ensureHistoryStartsWithUser(chatHistory);
      setSSEHeaders(res);

      // Check if we need to generate a title (look in 'chats' collection now)
      const publicChatRef = db.collection("chats").doc(sessionId);
      const publicChatDoc = await publicChatRef.get();
      // Fallback to 'sessions' for backward compatibility if needed, but focusing on new structure
      const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";

      let fullText = '';
      const streamingChat = model.startChat({
        history: chatHistory,
        generationConfig: { temperature: 0.7 },
      });

      const result = await streamingChat.sendMessageStream(message);
      
      for await (const chunk of result.stream) {
        const chunkText = chunk.text();
        if (chunkText) {
          fullText += chunkText;
          res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
        }
      }

      // Generate title if needed
      let newSessionName = null;
      if (isNewChat) {
        try {
          const titleModel = getGenerativeModel("You are a helpful assistant that generates short, concise titles for chat sessions.");
          const titleResult = await titleModel.generateContent(`Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`);
          const title = titleResult.response.text().trim();
          if (title) {
            await publicChatRef.update({ name: title });
            newSessionName = title;
          }
        } catch (error) {
          console.error("Error generating title:", error);
        }
      }

      const updatedHistory = [
        ...chatHistory.map((h: any) => ({ role: h.role, content: h.parts[0].text })),
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

