import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import crypto from "crypto";
import { genkit } from "genkit/beta";
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
class FirestoreSessionStore {
    async get(sessionId) {
        try {
            const sessionDoc = await db.collection("genkit_sessions").doc(sessionId).get();
            if (!sessionDoc.exists) {
                return undefined;
            }
            const data = sessionDoc.data();
            const sessionData = {
                id: sessionId,
                state: (data?.state || {}),
            };
            if (data?.history) {
                sessionData.history = data.history;
                console.log("Loaded session with history:", {
                    sessionId,
                    messageCount: Array.isArray(data.history) ? data.history.length : 0,
                });
            }
            else if (data?.threads) {
                const threads = data.threads;
                const threadKeys = Object.keys(threads);
                const firstThread = threadKeys.length > 0 ? threads[threadKeys[0]] : [];
                sessionData.history = Array.isArray(firstThread) ? firstThread : [];
                console.log("Migrated threads to history format:", {
                    sessionId,
                    messageCount: sessionData.history.length,
                });
            }
            else {
                sessionData.history = [];
                console.log("Loaded session without history:", sessionId);
            }
            return sessionData;
        }
        catch (error) {
            console.error("Error loading session:", error);
            return undefined;
        }
    }
    async save(sessionId, data) {
        try {
            const saveData = {
                id: sessionId,
                state: data.state || {},
                updatedAt: new Date(),
            };
            const dataAny = data;
            if (dataAny.history) {
                saveData.history = dataAny.history;
            }
            console.log("Saving session with history:", {
                sessionId,
                messageCount: Array.isArray(dataAny.history) ? dataAny.history.length : 0,
            });
            await db.collection("genkit_sessions").doc(sessionId).set(saveData, { merge: true });
            console.log("Session saved successfully with history");
        }
        catch (error) {
            console.error("Error saving session:", error);
            throw error;
        }
    }
}
const sessionStore = new FirestoreSessionStore();
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
    region: "us-central1",
    secrets: ["OPENAI_API_KEY"],
};
function okJson(res, body) {
    res.status(200).json(body);
}
function badRequest(res, message) {
    sendErrorResponse(res, 400, message);
}
/**
 * POST /focus/start
 * Body: { userId: string, courseId?: string, sessionId?: string }
 */
export const focusStart = onRequest(FUNCTION_CONFIG, async (req, res) => {
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
    }
    catch (error) {
        sendServerError(res, error);
    }
});
/**
 * POST /focus/stop
 * Body: { userId: string, focusSessionId: string }
 */
export const focusStop = onRequest(FUNCTION_CONFIG, async (req, res) => {
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
        const fs = fsDoc.data();
        if (fs.userId !== userId) {
            sendErrorResponse(res, 403, "Not allowed");
            return;
        }
        await fsRef.set({ status: "ended", endedAt: new Date(), updatedAt: new Date() }, { merge: true });
        okJson(res, { ok: true });
    }
    catch (error) {
        sendServerError(res, error);
    }
});
function convertHistoryToMessages(history) {
    const out = [];
    for (const h of history) {
        if (!h || typeof h.content !== "string")
            continue;
        const content = h.content.trim();
        if (!content)
            continue;
        const role = h.role === "user" ? "user" : "model";
        out.push({ role, content });
    }
    return out;
}
function toGenkitMessages(history, userMessage) {
    const messages = history.map((h) => ({
        role: h.role,
        content: [{ text: h.content }],
    }));
    messages.push({
        role: "user",
        content: [{ text: userMessage }],
    });
    return messages;
}
function setSSEHeaders(res) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
}
function sendErrorResponse(res, status, message) {
    res.status(status).json({ error: message });
}
function sendServerError(res, error) {
    console.error("Error:", error);
    res.status(500).json({
        error: error instanceof Error ? error.message : "Internal server error",
    });
}
async function loadSessionOrError(sessionId, res) {
    const session = await ai.loadSession(sessionId, { store: sessionStore });
    if (!session) {
        sendErrorResponse(res, 404, "Session not found. The session may have been deleted or never created.");
        return null;
    }
    return session;
}
export const chat = onRequest(FUNCTION_CONFIG, async (req, res) => {
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
            await sessionStore.save(sessionId, { id: sessionId, state: {}, history: [] });
            session = await ai.loadSession(sessionId, { store: sessionStore });
        }
        // If it still fails, treat as server error (store issues, etc).
        if (!session) {
            sendErrorResponse(res, 500, "Failed to initialize session");
            return;
        }
        const sessionDoc = await db.collection("genkit_sessions").doc(session.id).get();
        const sessionData = sessionDoc.data();
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
            onChunk: (chunk) => {
                const chunkText = typeof chunk.text === "string"
                    ? chunk.text
                    : typeof chunk.text === "function"
                        ? chunk.text()
                        : "";
                if (chunkText) {
                    fullText += chunkText;
                    res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
                }
            },
        });
        const finalText = typeof finalResponse.text === "string"
            ? finalResponse.text
            : typeof finalResponse.text === "function"
                ? finalResponse.text()
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
                });
                const rawTitle = typeof titleResult.text === "string"
                    ? titleResult.text
                    : typeof titleResult.text === "function"
                        ? titleResult.text()
                        : "";
                const title = rawTitle.trim().replace(/^['"]+|['"]+$/g, "").trim();
                if (title) {
                    await publicChatRef.update({ name: title });
                    newSessionName = title;
                }
            }
            catch (error) {
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
        });
        res.write(`data: ${JSON.stringify({ text: '', done: true, model: MODEL_NAME, sessionId: session.id, fullText, newSessionName })}\n\n`);
        res.end();
    }
    catch (error) {
        sendServerError(res, error);
    }
});
export const createChat = onRequest(FUNCTION_CONFIG, async (req, res) => {
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
    }
    catch (error) {
        sendServerError(res, error);
    }
});
export const getSessionHistory = onRequest(FUNCTION_CONFIG, async (req, res) => {
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
        if (!session)
            return;
        const sessionData = await sessionStore.get(sessionId);
        res.json({ state: sessionData?.state || {} });
    }
    catch (error) {
        sendServerError(res, error);
    }
});
//# sourceMappingURL=index.js.map