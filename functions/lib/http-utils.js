export function okJson(res, body) {
    res.status(200).json(body);
}
export function sendErrorResponse(res, status, message) {
    res.status(status).json({ error: message });
}
export function badRequest(res, message) {
    sendErrorResponse(res, 400, message);
}
export function sendServerError(res, error) {
    console.error("Error:", error);
    res.status(500).json({
        error: error instanceof Error ? error.message : "Internal server error",
    });
}
export function setSSEHeaders(res) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "*");
}
export function asRequiredString(value) {
    return typeof value === "string" && value.length > 0 ? value : null;
}
export function asOptionalString(value) {
    return typeof value === "string" ? value : null;
}
//# sourceMappingURL=http-utils.js.map