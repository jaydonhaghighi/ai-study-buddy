import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import crypto from "crypto";
import { genkit } from "genkit/beta";
import { openAI } from "@genkit-ai/compat-oai/openai";
import { FirestoreSessionStore } from "./firestore-session-store.js";
import { fromThreadMessages, toGenkitMessages, toThreadMessages } from "./chat-history.js";
import { asOptionalString, asRequiredString, badRequest, okJson, sendErrorResponse, sendServerError, setSSEHeaders, } from "./http-utils.js";
initializeApp();
const db = getFirestore();
const MAIN_THREAD = "main";
const MODEL_NAME = process.env.OPENAI_MODEL || "gpt-4o-mini";
const CHAT_MODEL = openAI.model(MODEL_NAME);
const ai = genkit({
    plugins: [openAI({
            apiKey: process.env.OPENAI_API_KEY || "",
        })],
    model: CHAT_MODEL,
});
const sessionStore = new FirestoreSessionStore(db, MAIN_THREAD);
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
        const body = req.body || {};
        const userId = asRequiredString(body.userId);
        const courseId = asOptionalString(body.courseId);
        const sessionId = asOptionalString(body.sessionId);
        if (!userId) {
            badRequest(res, "Missing required field: userId");
            return;
        }
        const focusSessionId = crypto.randomUUID();
        await db.collection("focusSessions").doc(focusSessionId).set({
            id: focusSessionId,
            userId,
            source: "webcam",
            status: "active",
            courseId: courseId ?? null,
            sessionId: sessionId ?? null,
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
        const body = req.body || {};
        const userId = asRequiredString(body.userId);
        const focusSessionId = asRequiredString(body.focusSessionId);
        if (!userId) {
            badRequest(res, "Missing required field: userId");
            return;
        }
        if (!focusSessionId) {
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
        if (!fs || fs.userId !== userId) {
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
export const chat = onRequest(FUNCTION_CONFIG, async (req, res) => {
    if (req.method !== "POST") {
        sendErrorResponse(res, 405, "Method not allowed");
        return;
    }
    try {
        const body = req.body || {};
        const sessionId = asRequiredString(body.sessionId);
        const message = asRequiredString(body.message);
        const userId = asRequiredString(body.userId);
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
        const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";
        let fullText = "";
        const modelMessages = toGenkitMessages(chatHistory, message);
        const generationOptions = {
            model: CHAT_MODEL,
            system: SYSTEM_INSTRUCTION,
            messages: modelMessages,
            config: { temperature: 0.7 },
            onChunk: (chunk) => {
                const chunkText = chunk.text;
                if (chunkText) {
                    fullText += chunkText;
                    res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
                }
            },
        };
        const finalResponse = await ai.generate(generationOptions);
        const finalText = finalResponse.text;
        if (typeof finalText === "string" && finalText.length >= fullText.length) {
            fullText = finalText;
        }
        // Generate title if needed
        let newSessionName = null;
        if (isNewChat) {
            try {
                const titleOptions = {
                    model: CHAT_MODEL,
                    system: "You are a helpful assistant that generates short, concise titles for chat sessions.",
                    prompt: `Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`,
                    config: { temperature: 0.2, maxOutputTokens: 24 },
                };
                const titleResult = await ai.generate(titleOptions);
                const rawTitle = titleResult.text;
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
            state: sessionData?.state ?? {},
            threads: { [MAIN_THREAD]: toThreadMessages(updatedHistory) },
        });
        res.write(`data: ${JSON.stringify({ text: '', done: true, model: MODEL_NAME, sessionId: session.id, fullText, newSessionName })}\n\n`);
        res.end();
    }
    catch (error) {
        sendServerError(res, error);
    }
});
//# sourceMappingURL=index.js.map