import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { genkit } from "genkit/beta";
import { googleAI } from "@genkit-ai/googleai";
import { GoogleGenerativeAI } from "@google/generative-ai";
initializeApp();
const db = getFirestore();
const ai = genkit({
    plugins: [googleAI({
            apiKey: process.env.GOOGLE_GENAI_API_KEY || "",
        })],
    model: googleAI.model("gemini-2.5-flash"),
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
            if (data?.threads) {
                sessionData.threads = data.threads;
                console.log("Loaded session with threads:", {
                    sessionId,
                    threadCount: Object.keys(data.threads).length,
                });
            }
            else {
                console.log("Loaded session without threads:", sessionId);
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
            if (dataAny.threads) {
                saveData.threads = dataAny.threads;
            }
            console.log("Saving session with threads:", {
                sessionId,
                hasThreads: !!dataAny.threads,
                threadCount: dataAny.threads ? Object.keys(dataAny.threads).length : 0,
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
export const chat = onRequest({
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
}, async (req, res) => {
    if (req.method !== "POST") {
        res.status(405).json({ error: "Method not allowed" });
        return;
    }
    try {
        const { sessionId, message, userId } = req.body;
        if (!sessionId || !message || !userId) {
            res.status(400).json({ error: "Missing required fields: sessionId, message, userId" });
            return;
        }
        console.log("Loading session:", sessionId);
        let session = await ai.loadSession(sessionId, {
            store: sessionStore,
        });
        if (!session) {
            console.error("Session not found:", sessionId);
            res.status(404).json({
                error: "Session not found. The session may have been deleted or never created.",
                sessionId: sessionId
            });
            return;
        }
        console.log("Loaded existing session:", session.id);
        const chat = session.chat({
            model: googleAI.model("gemini-2.5-flash"),
            system: "You are a helpful AI study buddy. You help students learn, answer questions, and provide educational support. Be friendly, encouraging, and clear in your explanations. Format your responses using Markdown for better readability: use **bold** for emphasis, `code blocks` for code examples, numbered or bulleted lists for steps, and headings for organizing information.",
        });
        const apiKey = process.env.GOOGLE_GENAI_API_KEY || "";
        const genAI = new GoogleGenerativeAI(apiKey);
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        const sessionDoc = await db.collection("genkit_sessions").doc(session.id).get();
        const sessionData = sessionDoc.data();
        const threads = sessionData?.threads || {};
        const threadKeys = Object.keys(threads);
        const history = threadKeys.length > 0 ? threads[threadKeys[0]] : [];
        const chatHistory = history.map((h) => ({
            role: h.role === "user" ? "user" : "model",
            parts: [{ text: h.content || h.text || "" }],
        }));
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*');
        let fullText = '';
        const streamingChat = model.startChat({
            history: chatHistory,
            generationConfig: {
                temperature: 0.7,
            },
        });
        const result = await streamingChat.sendMessageStream(message);
        for await (const chunk of result.stream) {
            const chunkText = chunk.text();
            if (chunkText) {
                fullText += chunkText;
                res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
            }
        }
        await chat.send(message);
        console.log("Message sent, session should be auto-saved by Genkit");
        res.write(`data: ${JSON.stringify({ text: '', done: true, model: "gemini-2.0-flash-exp", sessionId: session.id, fullText: fullText })}\n\n`);
        res.end();
    }
    catch (error) {
        console.error("Chat error:", error);
        res.status(500).json({
            error: error instanceof Error ? error.message : "Internal server error",
        });
    }
});
export const createSession = onRequest({
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
}, async (req, res) => {
    if (req.method !== "POST") {
        res.status(405).json({ error: "Method not allowed" });
        return;
    }
    try {
        const { userId, sessionName } = req.body;
        if (!userId) {
            res.status(400).json({ error: "Missing required field: userId" });
            return;
        }
        const session = ai.createSession({
            store: sessionStore,
        });
        console.log("Created new Genkit session:", session.id);
        await sessionStore.save(session.id, {
            id: session.id,
            state: {},
        });
        console.log("Session saved to Firestore:", session.id);
        await db.collection("sessions").doc(session.id).set({
            id: session.id,
            name: sessionName || `Chat ${Date.now()}`,
            userId: userId,
            createdAt: new Date(),
            lastMessageAt: new Date(),
        });
        res.json({
            sessionId: session.id,
            name: sessionName || `Chat ${Date.now()}`,
        });
    }
    catch (error) {
        console.error("Create session error:", error);
        res.status(500).json({
            error: error instanceof Error ? error.message : "Internal server error",
        });
    }
});
export const getSessionHistory = onRequest({
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
}, async (req, res) => {
    if (req.method !== "GET") {
        res.status(405).json({ error: "Method not allowed" });
        return;
    }
    try {
        const { sessionId } = req.query;
        if (!sessionId || typeof sessionId !== "string") {
            res.status(400).json({ error: "Missing required query parameter: sessionId" });
            return;
        }
        const session = await ai.loadSession(sessionId, {
            store: sessionStore,
        });
        if (!session) {
            res.status(404).json({ error: "Session not found" });
            return;
        }
        const sessionData = await sessionStore.get(sessionId);
        res.json({
            state: sessionData?.state || {},
        });
    }
    catch (error) {
        console.error("Get history error:", error);
        res.status(500).json({
            error: error instanceof Error ? error.message : "Internal server error",
        });
    }
});
//# sourceMappingURL=index.js.map