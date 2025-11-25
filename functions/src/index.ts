import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { genkit, SessionStore, SessionData } from "genkit/beta";
import { googleAI } from "@genkit-ai/googleai";

initializeApp();
const db = getFirestore();

const ai = genkit({
  plugins: [googleAI({
    apiKey: process.env.GOOGLE_GENAI_API_KEY || "",
  })],
  model: googleAI.model("gemini-1.5-flash"),
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
      return {
        id: sessionId,
        state: (data?.state || {}) as S,
      };
    } catch (error) {
      console.error("Error loading session:", error);
      return undefined;
    }
  }

  async save(sessionId: string, data: SessionData<S>): Promise<void> {
    try {
      await db.collection("genkit_sessions").doc(sessionId).set({
        id: sessionId,
        state: data.state || {},
        updatedAt: new Date(),
      }, { merge: true });
    } catch (error) {
      console.error("Error saving session:", error);
      throw error;
    }
  }
}

const sessionStore = new FirestoreSessionStore<StudyBuddyState>();

export const chat = onRequest(
  {
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
  },
  async (req, res) => {
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

      let session = await ai.loadSession(sessionId, {
        store: sessionStore,
      });

      if (!session) {
        session = ai.createSession({
          store: sessionStore,
        });
      }

      const chat = session.chat({
        system: "You are a helpful AI study buddy. You help students learn, answer questions, and provide educational support. Be friendly, encouraging, and clear in your explanations.",
      });

      const { text } = await chat.send(message);

      res.json({
        text: text,
        model: "gemini-1.5-flash",
        sessionId: session.id,
      });
    } catch (error) {
      console.error("Chat error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Internal server error",
      });
    }
  }
);

export const createSession = onRequest(
  {
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
  },
  async (req, res) => {
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
    } catch (error) {
      console.error("Create session error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Internal server error",
      });
    }
  }
);

export const getSessionHistory = onRequest(
  {
    cors: true,
    region: "us-central1",
    secrets: ["GOOGLE_GENAI_API_KEY"],
  },
  async (req, res) => {
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
    } catch (error) {
      console.error("Get history error:", error);
      res.status(500).json({
        error: error instanceof Error ? error.message : "Internal server error",
      });
    }
  }
);

