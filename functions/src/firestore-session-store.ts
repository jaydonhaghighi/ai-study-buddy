import type { Firestore } from "firebase-admin/firestore";
import type { SessionStore, SessionData } from "genkit/beta";
import type { MessageData } from "genkit";

export interface StudyBuddyState {
  userName?: string;
  preferences?: {
    learningStyle?: string;
    subject?: string;
  };
}

export class FirestoreSessionStore<S = unknown> implements SessionStore<S> {
  constructor(
    private readonly db: Firestore,
    private readonly mainThread: string
  ) {}

  async get(sessionId: string): Promise<SessionData<S> | undefined> {
    try {
      const sessionDoc = await this.db.collection("genkit_sessions").doc(sessionId).get();
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
        state: (data?.state ?? {}) as S,
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
        sessionData.threads = { [this.mainThread]: legacyThread };
      }

      console.log("Loaded session with history:", {
        sessionId,
        messageCount: sessionData.threads?.[this.mainThread]?.length ?? 0,
      });

      return sessionData;
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
        threads: data.threads || { [this.mainThread]: [] },
        updatedAt: new Date(),
      };
      console.log("Saving session with history:", {
        sessionId,
        messageCount: saveData.threads?.[this.mainThread]?.length ?? 0,
      });
      await this.db.collection("genkit_sessions").doc(sessionId).set(saveData, { merge: true });
      console.log("Session saved successfully with history");
    } catch (error) {
      console.error("Error saving session:", error);
      throw error;
    }
  }
}
