export function fromThreadMessages(messages) {
    const out = [];
    for (const message of messages || []) {
        if (message.role !== "user" && message.role !== "model")
            continue;
        const content = message.content
            .map((part) => (typeof part.text === "string" ? part.text : ""))
            .join("")
            .trim();
        if (!content)
            continue;
        out.push({ role: message.role, content });
    }
    return out;
}
export function toThreadMessages(history) {
    return history.map((h) => ({
        role: h.role,
        content: [{ text: h.content }],
    }));
}
export function toGenkitMessages(history, userMessage) {
    const messages = toThreadMessages(history);
    messages.push({
        role: "user",
        content: [{ text: userMessage }],
    });
    return messages;
}
//# sourceMappingURL=chat-history.js.map