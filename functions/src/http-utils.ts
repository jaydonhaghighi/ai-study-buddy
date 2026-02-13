export function okJson(res: any, body: any) {
  res.status(200).json(body);
}

export function sendErrorResponse(res: any, status: number, message: string) {
  res.status(status).json({ error: message });
}

export function badRequest(res: any, message: string) {
  sendErrorResponse(res, 400, message);
}

export function sendServerError(res: any, error: unknown) {
  console.error("Error:", error);
  res.status(500).json({
    error: error instanceof Error ? error.message : "Internal server error",
  });
}

export function setSSEHeaders(res: any) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("Access-Control-Allow-Origin", "*");
}

export function asRequiredString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function asOptionalString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}
