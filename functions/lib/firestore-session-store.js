export class FirestoreSessionStore {
    constructor(db, mainThread) {
        this.db = db;
        this.mainThread = mainThread;
    }
    async get(sessionId) {
        try {
            const sessionDoc = await this.db.collection("genkit_sessions").doc(sessionId).get();
            if (!sessionDoc.exists) {
                return undefined;
            }
            const data = sessionDoc.data();
            const sessionData = {
                id: sessionId,
                state: (data?.state ?? {}),
            };
            if (data?.threads && typeof data.threads === "object") {
                sessionData.threads = data.threads;
            }
            // Back-compat: older sessions stored flat "history".
            if (!sessionData.threads) {
                const legacyThread = Array.isArray(data?.history)
                    ? data.history
                        .filter((m) => !!m)
                        .map((m) => {
                        const role = m.role === "user" ? "user" : "model";
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
                state: data.state ?? {},
                threads: data.threads || { [this.mainThread]: [] },
                updatedAt: new Date(),
            };
            console.log("Saving session with history:", {
                sessionId,
                messageCount: saveData.threads?.[this.mainThread]?.length ?? 0,
            });
            await this.db.collection("genkit_sessions").doc(sessionId).set(saveData, { merge: true });
            console.log("Session saved successfully with history");
        }
        catch (error) {
            console.error("Error saving session:", error);
            throw error;
        }
    }
}
//# sourceMappingURL=firestore-session-store.js.map