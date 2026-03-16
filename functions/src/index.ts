import { onRequest } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { getFirestore, type Firestore } from "firebase-admin/firestore";
import { getStorage } from "firebase-admin/storage";
import crypto from "crypto";
import { genkit } from "genkit/beta";
import { openAI } from "@genkit-ai/compat-oai/openai";
import type {
  GenerateOptions,
  GenerateResponse,
  GenerateResponseChunk,
} from "genkit";
import { PDFParse } from "pdf-parse";
import mammoth from "mammoth";
import XLSX from "xlsx";
import JSZip from "jszip";
import { FirestoreSessionStore, type StudyBuddyState } from "./firestore-session-store.js";
import { fromThreadMessages, toGenkitMessages, toThreadMessages, type ChatHistoryMessage } from "./chat-history.js";
import {
  asOptionalString,
  asRequiredString,
  badRequest,
  okJson,
  sendErrorResponse,
  sendServerError,
  setSSEHeaders,
} from "./http-utils.js";

initializeApp();
const db = getFirestore();
const storage = getStorage();
const MAIN_THREAD = "main";

const MODEL_NAME = process.env.OPENAI_MODEL || "gpt-4o-mini";
const CHAT_MODEL = openAI.model(MODEL_NAME);
const OCR_MODEL_NAME = process.env.OPENAI_OCR_MODEL || MODEL_NAME;

const ai = genkit({
  plugins: [openAI({
    apiKey: process.env.OPENAI_API_KEY || "",
  })],
  model: CHAT_MODEL,
});

const sessionStore = new FirestoreSessionStore<StudyBuddyState>(db, MAIN_THREAD);

type StudyCoachMode = "nudge" | "recap";
type MaterialFileType = "pdf" | "docx" | "spreadsheet" | "slides" | "txt" | "image";
type MaterialLocationType = "page" | "sheet" | "slide" | "line" | "image";
type StudyDifficulty = "easy" | "medium" | "hard";
type StudySourceType = "chat" | "material";
type StudyQuizQuestionType = "mcq" | "short";
type FlashcardReviewRating = "again" | "hard" | "good" | "easy";

type CourseMaterialDoc = {
  id: string;
  userId: string;
  courseId?: string | null;
  sessionId?: string | null;
  chatId?: string | null;
  fileName: string;
  extension?: string | null;
  mimeType?: string | null;
  storagePath: string;
  fileType?: MaterialFileType;
  status?: string;
};

type MaterialChunkDoc = {
  id: string;
  materialId: string;
  userId: string;
  courseId: string | null;
  sessionId: string | null;
  chatId: string | null;
  fileName: string;
  fileType: MaterialFileType;
  locationType: MaterialLocationType;
  locationLabel: string;
  chunkIndex: number;
  text: string;
  textLower: string;
  createdAt: Date;
};

type ExtractedSegment = {
  text: string;
  locationType: MaterialLocationType;
  locationLabel: string;
};

type RagCitation = {
  id: string;
  materialId: string;
  fileName: string;
  fileType: MaterialFileType;
  locationType: MaterialLocationType;
  locationLabel: string;
  snippet: string;
  score: number;
  contextText: string;
};

type StudySourceDoc = {
  id: string;
  type: StudySourceType;
  label: string;
  snippet: string;
};

type StudyQuizQuestionDoc = {
  id: string;
  questionType: StudyQuizQuestionType;
  prompt: string;
  options: string[];
  correctAnswer: string;
  correctOptionIndex: number | null;
  explanation: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

type StudyFlashcardDoc = {
  id: string;
  front: string;
  back: string;
  tags: string[];
  difficulty: StudyDifficulty;
  sourceIds: string[];
  nextReviewAt: Date;
  intervalDays: number;
  easeFactor: number;
  repetitions: number;
  lastReviewedAt: Date | null;
};

type StudyExamQuestionDoc = {
  id: string;
  prompt: string;
  rubric: string[];
  modelAnswer: string;
  difficulty: StudyDifficulty;
  sourceIds: string[];
};

type StudySetDoc = {
  id: string;
  userId: string;
  chatId: string;
  courseId: string | null;
  sessionId: string | null;
  status: "generating" | "ready" | "failed";
  quizQuestions: StudyQuizQuestionDoc[];
  flashcards: StudyFlashcardDoc[];
  examQuestions: StudyExamQuestionDoc[];
  sources: StudySourceDoc[];
  model: string;
  generationMs: number | null;
  errorMessage: string | null;
  createdAt: Date;
  updatedAt: Date;
};

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
  secrets: ["OPENAI_API_KEY"],
};

const MAX_SOURCE_FILE_SIZE_BYTES = 25 * 1024 * 1024;
const MAX_RETRIEVAL_CHUNKS = 6;
const CHUNK_MAX_LEN = 1100;
const CHUNK_OVERLAP = 140;
const MAX_CHAT_MESSAGES_FOR_STUDY_SET = 30;
const MAX_STUDY_TRANSCRIPT_CHARS = 14000;
const DEFAULT_QUIZ_COUNT = 10;
const DEFAULT_FLASHCARD_COUNT = 14;
const DEFAULT_EXAM_COUNT = 4;

const SUPPORTED_EXTENSIONS = new Set(["pdf", "docx", "xlsx", "xls", "pptx", "ppt", "txt", "png", "jpg", "jpeg", "webp"]);

const STOPWORDS = new Set([
  "the", "and", "for", "with", "that", "this", "from", "have", "what", "when", "where", "which", "into", "your", "you",
  "about", "does", "are", "was", "were", "then", "than", "there", "their", "them", "been", "can", "could", "would",
  "should", "how", "why", "who", "all", "any", "each", "our", "its", "not", "but", "use", "using", "used",
]);

function asRequiredNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asOptionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clampInteger(input: number | null, fallback: number, min: number, max: number): number {
  if (input == null) return fallback;
  const floored = Math.floor(input);
  if (!Number.isFinite(floored)) return fallback;
  return Math.min(max, Math.max(min, floored));
}

function asDate(value: unknown): Date | null {
  if (value instanceof Date) return value;
  if (value && typeof value === "object" && "toDate" in value) {
    try {
      const out = (value as { toDate?: () => Date }).toDate?.();
      if (out instanceof Date) return out;
    } catch {
      return null;
    }
  }
  return null;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeDifficulty(value: unknown): StudyDifficulty {
  const candidate = asString(value).toLowerCase();
  if (candidate === "easy" || candidate === "medium" || candidate === "hard") return candidate;
  return "medium";
}

function normalizeSourceIds(value: unknown, allowed: Set<string>): string[] {
  if (!Array.isArray(value)) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const row of value) {
    const id = asString(row);
    if (!id || !allowed.has(id) || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
    if (out.length >= 4) break;
  }
  return out;
}

function extractJsonText(raw: string): string {
  const trimmed = raw.trim();
  const fenceMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenceMatch && typeof fenceMatch[1] === "string") {
    return fenceMatch[1].trim();
  }
  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }
  return trimmed;
}

function parseJsonObject(raw: string): Record<string, unknown> {
  const jsonText = extractJsonText(raw);
  const parsed = JSON.parse(jsonText);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Study generation response must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

function getFileExtension(fileName: string | null | undefined): string {
  if (!fileName) return "";
  const idx = fileName.lastIndexOf(".");
  if (idx < 0 || idx === fileName.length - 1) return "";
  return fileName.slice(idx + 1).toLowerCase();
}

function inferFileType(fileName: string, mimeType: string | null | undefined): MaterialFileType | null {
  const ext = getFileExtension(fileName);
  if (ext === "pdf" || mimeType === "application/pdf") return "pdf";
  if (ext === "docx" || mimeType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") return "docx";
  if (ext === "xlsx" || ext === "xls" || (mimeType || "").includes("spreadsheet") || mimeType === "application/vnd.ms-excel") return "spreadsheet";
  if (ext === "pptx" || ext === "ppt" || (mimeType || "").includes("presentation") || mimeType === "application/vnd.ms-powerpoint") return "slides";
  if (ext === "txt" || mimeType === "text/plain") return "txt";
  if (["png", "jpg", "jpeg", "webp"].includes(ext) || (mimeType || "").startsWith("image/")) return "image";
  return null;
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\r/g, "\n").replace(/\t/g, " ").replace(/[ \u00a0]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function truncateText(text: string, maxLength: number): string {
  const normalized = text.trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 1)}...`;
}

function decodeXmlEntities(input: string): string {
  return input
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_m, code: string) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_m, hex: string) => String.fromCharCode(Number.parseInt(hex, 16)));
}

function extractTextFromSlideXml(xml: string): string {
  const parts: string[] = [];
  const pattern = /<a:t[^>]*>([\s\S]*?)<\/a:t>/g;
  let match: RegExpExecArray | null = pattern.exec(xml);
  while (match) {
    parts.push(decodeXmlEntities(match[1]));
    match = pattern.exec(xml);
  }
  return normalizeWhitespace(parts.join("\n"));
}

function extractSlideNumber(path: string): number {
  const found = path.match(/slide(\d+)\.xml$/i);
  if (!found) return Number.MAX_SAFE_INTEGER;
  return Number.parseInt(found[1], 10);
}

async function extractTextFromImageWithOpenAI(buffer: Buffer, mimeType: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY || "";
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured for OCR");
  }

  const payload = {
    model: OCR_MODEL_NAME,
    messages: [
      {
        role: "system",
        content: "You extract text from images for study materials. Return plain text only. Do not add commentary.",
      },
      {
        role: "user",
        content: [
          { type: "text", text: "Extract all readable text from this image. Keep original wording as best as possible." },
          {
            type: "image_url",
            image_url: {
              url: `data:${mimeType};base64,${buffer.toString("base64")}`,
            },
          },
        ],
      },
    ],
    temperature: 0,
    max_tokens: 1200,
  };

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Image OCR failed (${response.status}): ${text || "unknown error"}`);
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | Array<{ type?: string; text?: string }> } }>;
  };

  const content = data.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return normalizeWhitespace(content);
  }
  if (Array.isArray(content)) {
    const txt = content
      .map((part) => (part?.type === "text" && typeof part.text === "string" ? part.text : ""))
      .join("\n");
    return normalizeWhitespace(txt);
  }

  return "";
}

async function extractSegmentsFromMaterial(buffer: Buffer, material: CourseMaterialDoc): Promise<{ fileType: MaterialFileType; segments: ExtractedSegment[] }> {
  const inferred = inferFileType(material.fileName, material.mimeType);
  if (!inferred) {
    throw new Error(`Unsupported material type: ${material.fileName}`);
  }

  if (!SUPPORTED_EXTENSIONS.has(getFileExtension(material.fileName))) {
    // Keep MIME-based support but reject unknown extensions for safety.
    if (!["pdf", "docx", "spreadsheet", "slides", "txt", "image"].includes(inferred)) {
      throw new Error(`Unsupported file extension for ${material.fileName}`);
    }
  }

  if (inferred === "pdf") {
    const parser = new PDFParse({ data: buffer });
    const parsed = await parser.getText();
    await parser.destroy();
    const rawText = typeof parsed.text === "string" ? parsed.text : "";
    const pages = rawText
      .split(/\f+/)
      .map((text: string) => normalizeWhitespace(text))
      .filter((text: string) => text.length > 0);

    if (pages.length > 1) {
      return {
        fileType: "pdf",
        segments: pages.map((text: string, idx: number) => ({
          text,
          locationType: "page",
          locationLabel: `Page ${idx + 1}`,
        })),
      };
    }

    const text = normalizeWhitespace(rawText);
    if (!text) throw new Error("No readable text found in PDF");
    return {
      fileType: "pdf",
      segments: [{ text, locationType: "page", locationLabel: "Page 1" }],
    };
  }

  if (inferred === "docx") {
    const result = await mammoth.extractRawText({ buffer });
    const text = normalizeWhitespace(result.value || "");
    if (!text) throw new Error("No readable text found in DOCX");
    return {
      fileType: "docx",
      segments: [{ text, locationType: "line", locationLabel: "Document" }],
    };
  }

  if (inferred === "spreadsheet") {
    const workbook = XLSX.read(buffer, { type: "buffer" });
    const segments: ExtractedSegment[] = [];

    for (const sheetName of workbook.SheetNames) {
      const sheet = workbook.Sheets[sheetName];
      if (!sheet) continue;

      const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, { header: 1, raw: false, defval: "" });
      const lines: string[] = [];
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex] || [];
        const values = row.map((cell) => String(cell).trim()).filter((cell) => cell.length > 0);
        if (values.length === 0) continue;
        lines.push(`Row ${rowIndex + 1}: ${values.join(" | ")}`);
      }

      const text = normalizeWhitespace(lines.join("\n"));
      if (text.length > 0) {
        segments.push({
          text,
          locationType: "sheet",
          locationLabel: `Sheet ${sheetName}`,
        });
      }
    }

    if (segments.length === 0) throw new Error("No readable text found in spreadsheet");
    return { fileType: "spreadsheet", segments };
  }

  if (inferred === "slides") {
    const zip = await JSZip.loadAsync(buffer);
    const slidePaths = Object.keys(zip.files)
      .filter((path) => /^ppt\/slides\/slide\d+\.xml$/i.test(path))
      .sort((a, b) => extractSlideNumber(a) - extractSlideNumber(b));

    const segments: ExtractedSegment[] = [];
    for (const slidePath of slidePaths) {
      // eslint-disable-next-line no-await-in-loop
      const slideXml = await zip.file(slidePath)?.async("string");
      if (!slideXml) continue;
      const text = extractTextFromSlideXml(slideXml);
      if (!text) continue;
      segments.push({
        text,
        locationType: "slide",
        locationLabel: `Slide ${extractSlideNumber(slidePath)}`,
      });
    }

    if (segments.length === 0) throw new Error("No readable text found in slide deck");
    return { fileType: "slides", segments };
  }

  if (inferred === "txt") {
    const text = normalizeWhitespace(buffer.toString("utf8"));
    if (!text) throw new Error("No readable text found in text file");
    return {
      fileType: "txt",
      segments: [{ text, locationType: "line", locationLabel: "Text" }],
    };
  }

  const mimeType = material.mimeType || "image/png";
  const text = await extractTextFromImageWithOpenAI(buffer, mimeType);
  if (!text) {
    throw new Error("No readable text found in image");
  }
  return {
    fileType: "image",
    segments: [{ text, locationType: "image", locationLabel: "Image OCR" }],
  };
}

function splitTextIntoChunks(text: string, maxLen: number, overlap: number): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return [];
  if (normalized.length <= maxLen) return [normalized];

  const chunks: string[] = [];
  let start = 0;

  while (start < normalized.length) {
    let end = Math.min(start + maxLen, normalized.length);

    if (end < normalized.length) {
      const minBoundary = Math.floor(start + maxLen * 0.6);
      const window = normalized.slice(minBoundary, end);
      const boundaryOffset = Math.max(window.lastIndexOf("\n"), window.lastIndexOf("."), window.lastIndexOf(" "));
      if (boundaryOffset > 0) {
        end = minBoundary + boundaryOffset + 1;
      }
    }

    const chunk = normalizeWhitespace(normalized.slice(start, end));
    if (chunk) {
      chunks.push(chunk);
    }

    if (end >= normalized.length) break;
    const nextStart = Math.max(0, end - overlap);
    if (nextStart <= start) {
      start = end;
    } else {
      start = nextStart;
    }
  }

  return chunks;
}

function buildChunkDocs(material: CourseMaterialDoc, fileType: MaterialFileType, segments: ExtractedSegment[]): MaterialChunkDoc[] {
  const out: MaterialChunkDoc[] = [];
  let chunkIndex = 0;

  for (const segment of segments) {
    const segmentChunks = splitTextIntoChunks(segment.text, CHUNK_MAX_LEN, CHUNK_OVERLAP);
    for (const text of segmentChunks) {
      chunkIndex += 1;
      const id = `${material.id}_${String(chunkIndex).padStart(5, "0")}`;
      out.push({
        id,
        materialId: material.id,
        userId: material.userId,
        courseId: material.courseId ?? null,
        sessionId: material.sessionId ?? null,
        chatId: material.chatId ?? null,
        fileName: material.fileName,
        fileType,
        locationType: segment.locationType,
        locationLabel: segment.locationLabel,
        chunkIndex,
        text,
        textLower: text.toLowerCase(),
        createdAt: new Date(),
      });
    }
  }

  return out;
}

async function deleteMaterialChunks(dbRef: Firestore, materialId: string): Promise<void> {
  while (true) {
    // eslint-disable-next-line no-await-in-loop
    const snapshot = await dbRef.collection("material_chunks").where("materialId", "==", materialId).limit(400).get();
    if (snapshot.empty) return;

    const batch = dbRef.batch();
    for (const doc of snapshot.docs) {
      batch.delete(doc.ref);
    }
    // eslint-disable-next-line no-await-in-loop
    await batch.commit();

    if (snapshot.size < 400) return;
  }
}

async function writeMaterialChunks(dbRef: Firestore, chunks: MaterialChunkDoc[]): Promise<void> {
  if (chunks.length === 0) return;

  for (let i = 0; i < chunks.length; i += 400) {
    const batch = dbRef.batch();
    const slice = chunks.slice(i, i + 400);
    for (const chunk of slice) {
      batch.set(dbRef.collection("material_chunks").doc(chunk.id), chunk);
    }
    // eslint-disable-next-line no-await-in-loop
    await batch.commit();
  }
}

function tokenize(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !STOPWORDS.has(token));
}

function scoreChunk(queryTerms: Set<string>, chunk: MaterialChunkDoc): number {
  if (queryTerms.size === 0) return 0;
  const chunkTokens = tokenize(chunk.textLower);
  if (chunkTokens.length === 0) return 0;

  let score = 0;
  const seen = new Set<string>();
  for (const token of chunkTokens) {
    if (!queryTerms.has(token)) continue;
    score += 1;
    if (!seen.has(token)) {
      seen.add(token);
      score += 0.6;
    }
  }

  return score;
}

async function retrieveRankedCitations(chatId: string, userId: string, userQuery: string): Promise<RagCitation[]> {
  const snapshot = await db.collection("material_chunks").where("chatId", "==", chatId).limit(1200).get();
  if (snapshot.empty) return [];

  const chunks = snapshot.docs
    .map((doc) => doc.data() as MaterialChunkDoc)
    .filter((chunk) => chunk.userId === userId && typeof chunk.text === "string" && chunk.text.trim().length > 0);

  if (chunks.length === 0) return [];

  const terms = new Set(tokenize(userQuery));
  const scored = chunks
    .map((chunk) => ({ chunk, score: scoreChunk(terms, chunk) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_RETRIEVAL_CHUNKS);

  if (scored.length === 0) return [];

  return scored.map((item, idx) => ({
    id: `C${idx + 1}`,
    materialId: item.chunk.materialId,
    fileName: item.chunk.fileName,
    fileType: item.chunk.fileType,
    locationType: item.chunk.locationType,
    locationLabel: item.chunk.locationLabel,
    snippet: truncateText(item.chunk.text, 260),
    score: Math.round(item.score * 100) / 100,
    contextText: truncateText(item.chunk.text, 900),
  }));
}

function buildRagUserPrompt(question: string, citations: RagCitation[]): string {
  if (citations.length === 0) return question;

  const context = citations
    .map((citation) => {
      return [
        `[${citation.id}] ${citation.fileName} (${citation.locationLabel})`,
        citation.contextText,
      ].join("\n");
    })
    .join("\n\n");

  return [
    "Use the context snippets below to answer the question.",
    "Rules:",
    "- Use only the provided snippets for factual claims.",
    "- If information is missing, clearly say you do not know from the uploaded material.",
    "- Cite supporting snippets inline using [C#] markers.",
    "",
    "Context snippets:",
    context,
    "",
    `Question: ${question}`,
  ].join("\n");
}

function selectFinalCitations(answerText: string, citations: RagCitation[]): RagCitation[] {
  if (citations.length === 0) return [];
  const used = new Set<string>();
  const regex = /\[(C\d+)\]/g;
  let match: RegExpExecArray | null = regex.exec(answerText);
  while (match) {
    used.add(match[1]);
    match = regex.exec(answerText);
  }

  if (used.size === 0) {
    return citations.slice(0, Math.min(3, citations.length));
  }

  return citations.filter((citation) => used.has(citation.id));
}

type ChatMessageForStudy = {
  id: string;
  text: string;
  isAI: boolean;
  createdAt: Date | null;
};

async function loadChatOwnership(chatId: string): Promise<{ userId: string; courseId: string | null; sessionId: string | null } | null> {
  const chatRef = db.collection("chats").doc(chatId);
  const chatDoc = await chatRef.get();
  if (!chatDoc.exists) return null;
  const data = chatDoc.data() as Record<string, unknown>;
  return {
    userId: typeof data.userId === "string" ? data.userId : "",
    courseId: typeof data.courseId === "string" ? data.courseId : null,
    sessionId: typeof data.sessionId === "string" ? data.sessionId : null,
  };
}

async function loadRecentChatMessages(chatId: string, userId: string, limitCount: number): Promise<ChatMessageForStudy[]> {
  const snapshot = await db.collection("messages")
    .where("sessionId", "==", chatId)
    .where("userId", "==", userId)
    .orderBy("createdAt", "desc")
    .limit(limitCount)
    .get();

  return snapshot.docs
    .map((doc) => {
      const data = doc.data() as Record<string, unknown>;
      const textField = typeof data.text === "string"
        ? data.text
        : (data.text && typeof data.text === "object" && "text" in data.text)
          ? asString((data.text as Record<string, unknown>).text)
          : "";
      return {
        id: doc.id,
        text: textField,
        isAI: !!data.isAI,
        createdAt: asDate(data.createdAt),
      } satisfies ChatMessageForStudy;
    })
    .sort((a, b) => {
      const aTime = a.createdAt ? a.createdAt.getTime() : 0;
      const bTime = b.createdAt ? b.createdAt.getTime() : 0;
      return aTime - bTime;
    });
}

function buildChatSourcesAndTranscript(messages: ChatMessageForStudy[]): { sources: StudySourceDoc[]; transcript: string } {
  const sources: StudySourceDoc[] = [];
  const lines: string[] = [];
  let idx = 0;

  for (const message of messages) {
    const normalizedText = normalizeWhitespace(message.text || "");
    if (!normalizedText) continue;
    idx += 1;
    const sourceId = `M${idx}`;
    const role = message.isAI ? "AI" : "You";
    sources.push({
      id: sourceId,
      type: "chat",
      label: `${role} message ${idx}`,
      snippet: truncateText(normalizedText, 220),
    });
    lines.push(`[${sourceId}] ${role}: ${truncateText(normalizedText, 420)}`);
  }

  let transcript = lines.join("\n");
  if (transcript.length > MAX_STUDY_TRANSCRIPT_CHARS) {
    transcript = transcript.slice(transcript.length - MAX_STUDY_TRANSCRIPT_CHARS);
  }

  return { sources, transcript };
}

function buildMaterialSources(citations: RagCitation[]): StudySourceDoc[] {
  return citations.map((citation) => ({
    id: citation.id,
    type: "material",
    label: `${citation.fileName} (${citation.locationLabel})`,
    snippet: truncateText(citation.contextText || citation.snippet, 220),
  }));
}

function buildStudyGenerationPrompt({
  quizCount,
  flashcardCount,
  examCount,
  transcript,
  sources,
}: {
  quizCount: number;
  flashcardCount: number;
  examCount: number;
  transcript: string;
  sources: StudySourceDoc[];
}): string {
  const sourceCatalog = sources
    .map((source) => `- ${source.id} [${source.type}] ${source.label}: ${source.snippet}`)
    .join("\n");

  return [
    "Generate active-recall study assets from the chat transcript and source catalog.",
    "Output only valid JSON (no markdown, no prose) with this top-level shape:",
    "{",
    '  "quiz": [',
    "    {",
    '      "type": "mcq" | "short",',
    '      "question": "string",',
    '      "choices": ["string"]  // required for mcq only, 4 options preferred,',
    '      "answerIndex": 0,        // required for mcq only, 0-based index into choices,',
    '      "answer": "string",     // required for short only,',
    '      "explanation": "string",',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["M1","C1"]',
    "    }",
    "  ],",
    '  "flashcards": [',
    "    {",
    '      "front": "string",',
    '      "back": "string",',
    '      "tags": ["string"],',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["M2","C2"]',
    "    }",
    "  ],",
    '  "examQuestions": [',
    "    {",
    '      "prompt": "string",',
    '      "rubric": ["criterion string"],',
    '      "modelAnswer": "string",',
    '      "difficulty": "easy" | "medium" | "hard",',
    '      "sourceIds": ["C1","M3"]',
    "    }",
    "  ]",
    "}",
    "",
    `Required counts: quiz=${quizCount}, flashcards=${flashcardCount}, examQuestions=${examCount}.`,
    "Rules:",
    "- Use only facts supported by transcript/source catalog.",
    "- Every item must include at least one source id from the catalog.",
    "- Mix difficulty levels across the set.",
    "- Keep wording concise and student-friendly.",
    "",
    "Source catalog:",
    sourceCatalog || "- none",
    "",
    "Chat transcript:",
    transcript || "(empty)",
  ].join("\n");
}

function parseStudyQuizQuestions(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
): StudyQuizQuestionDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyQuizQuestionDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const questionType: StudyQuizQuestionType = asString(data.type).toLowerCase() === "mcq" ? "mcq" : "short";
    const prompt = asString(data.question) || asString(data.prompt);
    if (prompt.length < 6) continue;

    const explanation = asString(data.explanation);
    const difficulty = normalizeDifficulty(data.difficulty);
    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    if (questionType === "mcq") {
      const options = Array.isArray(data.choices)
        ? data.choices.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 6)
        : [];
      if (options.length < 2) continue;

      let correctOptionIndex: number | null =
        typeof data.answerIndex === "number" && Number.isFinite(data.answerIndex)
          ? Math.floor(data.answerIndex)
          : null;
      if (correctOptionIndex == null || correctOptionIndex < 0 || correctOptionIndex >= options.length) {
        const answerText = asString(data.answer).toLowerCase();
        const matchIndex = answerText ? options.findIndex((option) => option.toLowerCase() === answerText) : -1;
        correctOptionIndex = matchIndex >= 0 ? matchIndex : 0;
      }

      out.push({
        id: `Q${out.length + 1}`,
        questionType: "mcq",
        prompt,
        options,
        correctAnswer: options[correctOptionIndex] || options[0],
        correctOptionIndex,
        explanation,
        difficulty,
        sourceIds: normalizedSourceIds,
      });
      continue;
    }

    const shortAnswer = asString(data.answer) || asString(data.correctAnswer);
    if (!shortAnswer) continue;
    out.push({
      id: `Q${out.length + 1}`,
      questionType: "short",
      prompt,
      options: [],
      correctAnswer: shortAnswer,
      correctOptionIndex: null,
      explanation,
      difficulty,
      sourceIds: normalizedSourceIds,
    });
  }

  return out;
}

function parseStudyFlashcards(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
  now: Date,
): StudyFlashcardDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyFlashcardDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const front = asString(data.front);
    const back = asString(data.back);
    if (front.length < 4 || back.length < 4) continue;

    const tags = Array.isArray(data.tags)
      ? data.tags.map((item) => asString(item)).filter((tag) => tag.length > 0).slice(0, 6)
      : [];

    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    out.push({
      id: `F${out.length + 1}`,
      front,
      back,
      tags,
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: normalizedSourceIds,
      nextReviewAt: new Date(now.getTime() + 24 * 60 * 60 * 1000),
      intervalDays: 1,
      easeFactor: 2.5,
      repetitions: 0,
      lastReviewedAt: null,
    });
  }

  return out;
}

function parseStudyExamQuestions(
  value: unknown,
  maxCount: number,
  allowedSourceIds: Set<string>,
): StudyExamQuestionDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyExamQuestionDoc[] = [];
  const fallbackSourceId = allowedSourceIds.values().next().value as string | undefined;

  for (const row of value) {
    if (out.length >= maxCount) break;
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const prompt = asString(data.prompt) || asString(data.question);
    if (prompt.length < 6) continue;

    const rubric = Array.isArray(data.rubric)
      ? data.rubric.map((item) => asString(item)).filter((item) => item.length > 0).slice(0, 6)
      : [];
    const modelAnswer = asString(data.modelAnswer) || asString(data.answer);
    if (!modelAnswer) continue;

    const sourceIds = normalizeSourceIds(data.sourceIds, allowedSourceIds);
    const normalizedSourceIds = sourceIds.length > 0
      ? sourceIds
      : (fallbackSourceId ? [fallbackSourceId] : []);

    out.push({
      id: `E${out.length + 1}`,
      prompt,
      rubric,
      modelAnswer,
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: normalizedSourceIds,
    });
  }

  return out;
}

function applyFlashcardRating(card: StudyFlashcardDoc, rating: FlashcardReviewRating, now: Date): StudyFlashcardDoc {
  let easeFactor = card.easeFactor || 2.5;
  let repetitions = Math.max(0, card.repetitions || 0);
  let intervalDays = Math.max(1, card.intervalDays || 1);

  if (rating === "again") {
    repetitions = 0;
    intervalDays = 1;
    easeFactor = Math.max(1.3, easeFactor - 0.2);
  } else if (rating === "hard") {
    repetitions += 1;
    intervalDays = repetitions <= 1 ? 1 : Math.max(2, Math.round(intervalDays * 1.2));
    easeFactor = Math.max(1.3, easeFactor - 0.15);
  } else if (rating === "good") {
    repetitions += 1;
    intervalDays = repetitions === 1 ? 1 : repetitions === 2 ? 3 : Math.max(2, Math.round(intervalDays * easeFactor));
  } else {
    repetitions += 1;
    intervalDays = repetitions === 1 ? 2 : repetitions === 2 ? 5 : Math.max(3, Math.round(intervalDays * easeFactor * 1.3));
    easeFactor = Math.max(1.3, easeFactor + 0.15);
  }

  const nextReviewAt = new Date(now.getTime() + intervalDays * 24 * 60 * 60 * 1000);
  return {
    ...card,
    easeFactor: Math.round(easeFactor * 100) / 100,
    repetitions,
    intervalDays,
    nextReviewAt,
    lastReviewedAt: now,
  };
}

function parseStoredFlashcards(value: unknown): StudyFlashcardDoc[] {
  if (!Array.isArray(value)) return [];
  const out: StudyFlashcardDoc[] = [];

  for (const row of value) {
    if (!row || typeof row !== "object") continue;
    const data = row as Record<string, unknown>;
    const front = asString(data.front);
    const back = asString(data.back);
    if (!front || !back) continue;

    out.push({
      id: asString(data.id) || `F${out.length + 1}`,
      front,
      back,
      tags: Array.isArray(data.tags)
        ? data.tags.map((item) => asString(item)).filter((tag) => tag.length > 0).slice(0, 6)
        : [],
      difficulty: normalizeDifficulty(data.difficulty),
      sourceIds: Array.isArray(data.sourceIds)
        ? data.sourceIds.map((item) => asString(item)).filter((id) => id.length > 0).slice(0, 6)
        : [],
      nextReviewAt: asDate(data.nextReviewAt) || new Date(),
      intervalDays: clampInteger(asOptionalNumber(data.intervalDays), 1, 1, 3650),
      easeFactor: Math.max(1.3, asOptionalNumber(data.easeFactor) ?? 2.5),
      repetitions: Math.max(0, clampInteger(asOptionalNumber(data.repetitions), 0, 0, 1000)),
      lastReviewedAt: asDate(data.lastReviewedAt),
    });
  }

  return out;
}

async function loadMaterialById(materialId: string): Promise<CourseMaterialDoc | null> {
  const ref = db.collection("courseMaterials").doc(materialId);
  const snap = await ref.get();
  if (!snap.exists) return null;
  const data = snap.data() as Record<string, unknown>;

  return {
    id: snap.id,
    userId: typeof data.userId === "string" ? data.userId : "",
    courseId: typeof data.courseId === "string" ? data.courseId : null,
    sessionId: typeof data.sessionId === "string" ? data.sessionId : null,
    chatId: typeof data.chatId === "string" ? data.chatId : null,
    fileName: typeof data.fileName === "string" ? data.fileName : snap.id,
    extension: typeof data.extension === "string" ? data.extension : null,
    mimeType: typeof data.mimeType === "string" ? data.mimeType : null,
    storagePath: typeof data.storagePath === "string" ? data.storagePath : "",
    fileType: typeof data.fileType === "string" ? (data.fileType as MaterialFileType) : undefined,
    status: typeof data.status === "string" ? data.status : undefined,
  };
}

async function buildChunksForMaterial(material: CourseMaterialDoc): Promise<{ fileType: MaterialFileType; chunks: MaterialChunkDoc[] }> {
  const file = storage.bucket().file(material.storagePath);
  const [exists] = await file.exists();
  if (!exists) {
    throw new Error("Uploaded file not found in storage");
  }

  const [buffer] = await file.download();
  if (buffer.byteLength > MAX_SOURCE_FILE_SIZE_BYTES) {
    throw new Error("File is too large to process (max 25MB)");
  }

  const extracted = await extractSegmentsFromMaterial(buffer, material);
  const chunks = buildChunkDocs(material, extracted.fileType, extracted.segments);
  if (chunks.length === 0) {
    throw new Error("Could not extract any indexable content from this file");
  }

  return { fileType: extracted.fileType, chunks };
}

/**
 * POST /focus/start
 * Body: { userId: string, courseId?: string, sessionId?: string }
 */
export const focusStart = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
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
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /focus/stop
 * Body: { userId: string, focusSessionId: string }
 */
export const focusStop = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
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
      const fs = fsDoc.data() as { userId?: string } | undefined;
      if (!fs || fs.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      await fsRef.set({ status: "ended", endedAt: new Date(), updatedAt: new Date() }, { merge: true });

      okJson(res, { ok: true });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /studyCoach
 */
export const studyCoach = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }
    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const mode = asRequiredString(body.mode) as StudyCoachMode | null;
      const phase = asRequiredString(body.phase);
      const eventType = asRequiredString(body.eventType);
      const sprintIndex = asRequiredNumber(body.sprintIndex);
      const elapsedSec = asRequiredNumber(body.elapsedSec);
      const remainingSec = asRequiredNumber(body.remainingSec);
      const focusPercent = asOptionalNumber(body.focusPercent);
      const distractionCount = asOptionalNumber(body.distractionCount);
      const firstDriftSec = asOptionalNumber(body.firstDriftSec);

      if (!userId) {
        badRequest(res, "Missing required field: userId");
        return;
      }
      if (!mode || (mode !== "nudge" && mode !== "recap")) {
        badRequest(res, "Missing or invalid required field: mode");
        return;
      }
      if (!phase) {
        badRequest(res, "Missing required field: phase");
        return;
      }
      if (!eventType) {
        badRequest(res, "Missing required field: eventType");
        return;
      }
      if (sprintIndex == null) {
        badRequest(res, "Missing required field: sprintIndex");
        return;
      }
      if (elapsedSec == null) {
        badRequest(res, "Missing required field: elapsedSec");
        return;
      }
      if (remainingSec == null) {
        badRequest(res, "Missing required field: remainingSec");
        return;
      }

      const instruction =
        mode === "nudge"
          ? "Write one short real-time coaching nudge. Keep it under 22 words."
          : "Write one short sprint recap. Mention performance and one concrete next goal in under 40 words.";

      const prompt = [
        instruction,
        "Tone: supportive, direct, non-judgmental, no emojis, no markdown.",
        `eventType=${eventType}`,
        `phase=${phase}`,
        `sprintIndex=${sprintIndex}`,
        `elapsedSec=${elapsedSec}`,
        `remainingSec=${remainingSec}`,
        `focusPercent=${focusPercent == null ? "unknown" : focusPercent.toFixed(1)}`,
        `distractionCount=${distractionCount == null ? "unknown" : distractionCount}`,
        `firstDriftSec=${firstDriftSec == null ? "unknown" : firstDriftSec}`,
      ].join("\n");

      const coachResponse: GenerateResponse = await ai.generate({
        model: CHAT_MODEL,
        system:
          "You are an AI study coach for Pomodoro sessions. Be concise, specific, and actionable. Return plain text only.",
        prompt,
        config: {
          temperature: 0.5,
          maxOutputTokens: mode === "nudge" ? 64 : 96,
        },
      });

      const rawMessage = coachResponse.text ?? "";
      const cleaned = rawMessage
        .replace(/\s+/g, " ")
        .replace(/^["']+|["']+$/g, "")
        .trim();

      const fallback =
        mode === "nudge"
          ? "Stay with this sprint. One small focused step right now."
          : "Solid effort this sprint. Keep the next sprint focused and aim to improve your focused minutes.";

      okJson(res, { ok: true, message: cleaned.length > 0 ? cleaned : fallback });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /materialIndex
 * Body: { userId: string, materialId: string }
 */
export const materialIndex = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const materialId = asRequiredString(body.materialId);

    if (!userId || !materialId) {
      badRequest(res, "Missing required fields: userId, materialId");
      return;
    }

    const materialRef = db.collection("courseMaterials").doc(materialId);

    try {
      const material = await loadMaterialById(materialId);
      if (!material) {
        sendErrorResponse(res, 404, "Material not found");
        return;
      }
      if (!material.userId || material.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }
      if (!material.storagePath) {
        sendErrorResponse(res, 400, "Material has no storagePath");
        return;
      }

      await materialRef.set(
        {
          status: "processing",
          errorMessage: null,
          updatedAt: new Date(),
        },
        { merge: true }
      );

      const startedAt = Date.now();
      await deleteMaterialChunks(db, material.id);
      const { fileType, chunks } = await buildChunksForMaterial(material);
      await writeMaterialChunks(db, chunks);

      await materialRef.set(
        {
          fileType,
          extension: getFileExtension(material.fileName),
          status: "indexed",
          chunkCount: chunks.length,
          processingMs: Date.now() - startedAt,
          errorMessage: null,
          updatedAt: new Date(),
        },
        { merge: true }
      );

      okJson(res, { ok: true, materialId, chunkCount: chunks.length, fileType });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown material indexing error";
      await materialRef.set(
        {
          status: "failed",
          errorMessage: message,
          updatedAt: new Date(),
        },
        { merge: true }
      );
      sendErrorResponse(res, 500, message);
    }
  }
);

/**
 * POST /materialDelete
 * Body: { userId: string, materialId: string }
 */
export const materialDelete = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    try {
      const body = req.body || {};
      const userId = asRequiredString(body.userId);
      const materialId = asRequiredString(body.materialId);
      if (!userId || !materialId) {
        badRequest(res, "Missing required fields: userId, materialId");
        return;
      }

      const material = await loadMaterialById(materialId);
      if (!material) {
        sendErrorResponse(res, 404, "Material not found");
        return;
      }
      if (material.userId !== userId) {
        sendErrorResponse(res, 403, "Not allowed");
        return;
      }

      await deleteMaterialChunks(db, material.id);

      if (material.storagePath) {
        try {
          await storage.bucket().file(material.storagePath).delete({ ignoreNotFound: true });
        } catch (error) {
          console.warn("Failed to delete storage object for material", { materialId, error });
        }
      }

      await db.collection("courseMaterials").doc(materialId).delete();
      okJson(res, { ok: true, materialId });
    } catch (error) {
      sendServerError(res, error);
    }
  }
);

/**
 * POST /studyGenerate
 * Body: { userId: string, chatId: string, quizCount?: number, flashcardCount?: number, examCount?: number }
 */
export const studyGenerate = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const chatId = asRequiredString(body.chatId);
    const quizCount = clampInteger(asOptionalNumber(body.quizCount), DEFAULT_QUIZ_COUNT, 4, 20);
    const flashcardCount = clampInteger(asOptionalNumber(body.flashcardCount), DEFAULT_FLASHCARD_COUNT, 6, 30);
    const examCount = clampInteger(asOptionalNumber(body.examCount), DEFAULT_EXAM_COUNT, 2, 10);

    if (!userId || !chatId) {
      badRequest(res, "Missing required fields: userId, chatId");
      return;
    }

    const ownership = await loadChatOwnership(chatId);
    if (!ownership) {
      sendErrorResponse(res, 404, "Chat not found");
      return;
    }
    if (ownership.userId !== userId) {
      sendErrorResponse(res, 403, "Not allowed");
      return;
    }

    const now = new Date();
    const studySetRef = db.collection("studySets").doc();

    await studySetRef.set({
      id: studySetRef.id,
      userId,
      chatId,
      courseId: ownership.courseId,
      sessionId: ownership.sessionId,
      status: "generating",
      quizQuestions: [],
      flashcards: [],
      examQuestions: [],
      sources: [],
      model: MODEL_NAME,
      generationMs: null,
      errorMessage: null,
      createdAt: now,
      updatedAt: now,
    } satisfies StudySetDoc);

    try {
      const recentMessages = await loadRecentChatMessages(chatId, userId, MAX_CHAT_MESSAGES_FOR_STUDY_SET);
      if (recentMessages.length === 0) {
        throw new Error("No chat messages found. Ask at least one question before generating a study set.");
      }

      const retrievalSeed = recentMessages
        .filter((msg) => !msg.isAI)
        .slice(-6)
        .map((msg) => msg.text)
        .join("\n");
      const ragCitations = await retrieveRankedCitations(chatId, userId, retrievalSeed || recentMessages[recentMessages.length - 1]?.text || "study topic");

      const { sources: chatSources, transcript } = buildChatSourcesAndTranscript(recentMessages);
      const materialSources = buildMaterialSources(ragCitations);
      const allSources = [...chatSources, ...materialSources];
      const allowedSourceIds = new Set(allSources.map((source) => source.id));

      const prompt = buildStudyGenerationPrompt({
        quizCount,
        flashcardCount,
        examCount,
        transcript,
        sources: allSources,
      });

      const startedAtMs = Date.now();
      const generation: GenerateResponse = await ai.generate({
        model: CHAT_MODEL,
        system: "You create source-grounded study materials. Return JSON only.",
        prompt,
        config: {
          temperature: 0.35,
          maxOutputTokens: 2600,
        },
      });

      const rawOutput = generation.text || "";
      if (!rawOutput.trim()) {
        throw new Error("Model returned an empty study set");
      }

      const parsed = parseJsonObject(rawOutput);
      const quizQuestions = parseStudyQuizQuestions(parsed.quiz, quizCount, allowedSourceIds);
      const flashcards = parseStudyFlashcards(parsed.flashcards, flashcardCount, allowedSourceIds, now);
      const examQuestions = parseStudyExamQuestions(parsed.examQuestions, examCount, allowedSourceIds);

      if (quizQuestions.length === 0 || flashcards.length === 0 || examQuestions.length === 0) {
        throw new Error("Generated study set was incomplete. Try again with a richer chat context.");
      }

      await studySetRef.set({
        status: "ready",
        quizQuestions,
        flashcards,
        examQuestions,
        sources: allSources,
        generationMs: Date.now() - startedAtMs,
        errorMessage: null,
        updatedAt: new Date(),
      }, { merge: true });

      okJson(res, {
        ok: true,
        studySetId: studySetRef.id,
        quizCount: quizQuestions.length,
        flashcardCount: flashcards.length,
        examCount: examQuestions.length,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Study set generation failed";
      await studySetRef.set({
        status: "failed",
        errorMessage: message,
        updatedAt: new Date(),
      }, { merge: true });
      sendErrorResponse(res, 500, message);
    }
  }
);

/**
 * POST /flashcardReview
 * Body: { userId: string, studySetId: string, cardId: string, rating: "again"|"hard"|"good"|"easy" }
 */
export const flashcardReview = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
    if (req.method !== "POST") {
      sendErrorResponse(res, 405, "Method not allowed");
      return;
    }

    const body = req.body || {};
    const userId = asRequiredString(body.userId);
    const studySetId = asRequiredString(body.studySetId);
    const cardId = asRequiredString(body.cardId);
    const ratingRaw = asRequiredString(body.rating);
    const validRatings: FlashcardReviewRating[] = ["again", "hard", "good", "easy"];
    const rating = validRatings.includes((ratingRaw || "") as FlashcardReviewRating)
      ? (ratingRaw as FlashcardReviewRating)
      : null;

    if (!userId || !studySetId || !cardId || !rating) {
      badRequest(res, "Missing required fields: userId, studySetId, cardId, rating");
      return;
    }

    const studySetRef = db.collection("studySets").doc(studySetId);
    const studySetSnap = await studySetRef.get();
    if (!studySetSnap.exists) {
      sendErrorResponse(res, 404, "Study set not found");
      return;
    }
    const data = studySetSnap.data() as Record<string, unknown>;
    if (asString(data.userId) !== userId) {
      sendErrorResponse(res, 403, "Not allowed");
      return;
    }

    const flashcards = parseStoredFlashcards(data.flashcards);
    const cardIndex = flashcards.findIndex((card) => card.id === cardId);
    if (cardIndex < 0) {
      sendErrorResponse(res, 404, "Flashcard not found");
      return;
    }

    const now = new Date();
    const updatedCard = applyFlashcardRating(flashcards[cardIndex], rating, now);
    flashcards[cardIndex] = updatedCard;

    await studySetRef.set({
      flashcards,
      updatedAt: now,
    }, { merge: true });

    await db.collection("flashcardReviews").add({
      userId,
      studySetId,
      chatId: asString(data.chatId) || null,
      cardId,
      rating,
      intervalDays: updatedCard.intervalDays,
      easeFactor: updatedCard.easeFactor,
      repetitions: updatedCard.repetitions,
      reviewedAt: now,
      createdAt: now,
    });

    okJson(res, {
      ok: true,
      cardId: updatedCard.id,
      nextReviewAt: updatedCard.nextReviewAt.toISOString(),
      intervalDays: updatedCard.intervalDays,
      easeFactor: updatedCard.easeFactor,
      repetitions: updatedCard.repetitions,
    });
  }
);

export const chat = onRequest(
  FUNCTION_CONFIG,
  async (req, res) => {
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

      let session = await ai.loadSession(sessionId, { store: sessionStore });
      if (!session) {
        await sessionStore.save(sessionId, {
          id: sessionId,
          state: {},
          threads: { [MAIN_THREAD]: [] },
        });
        session = await ai.loadSession(sessionId, { store: sessionStore });
      }
      if (!session) {
        sendErrorResponse(res, 500, "Failed to initialize session");
        return;
      }

      const sessionData = await sessionStore.get(session.id);
      const chatHistory = fromThreadMessages(sessionData?.threads?.[MAIN_THREAD]);
      setSSEHeaders(res);

      const publicChatRef = db.collection("chats").doc(sessionId);
      const publicChatDoc = await publicChatRef.get();
      const isNewChat = publicChatDoc.exists && publicChatDoc.data()?.name === "New Chat";

      const ragCitations = await retrieveRankedCitations(sessionId, userId, message);
      const ragPrompt = buildRagUserPrompt(message, ragCitations);

      let fullText = "";
      const modelMessages = toGenkitMessages(chatHistory, ragPrompt);
      const generationOptions: GenerateOptions = {
        model: CHAT_MODEL,
        system:
          ragCitations.length > 0
            ? `${SYSTEM_INSTRUCTION}\n\nWhen context snippets are included by the user message, use only those snippets for factual claims and cite them using [C#].`
            : `${SYSTEM_INSTRUCTION}\n\nIf the question requires course-specific source material and none is available, clearly state that and ask for upload.`,
        messages: modelMessages,
        config: { temperature: 0.5 },
        onChunk: (chunk: GenerateResponseChunk) => {
          const chunkText = chunk.text;
          if (chunkText) {
            fullText += chunkText;
            res.write(`data: ${JSON.stringify({ text: chunkText, done: false })}\n\n`);
          }
        },
      };
      const finalResponse: GenerateResponse = await ai.generate(generationOptions);
      const finalText = finalResponse.text;
      if (typeof finalText === "string" && finalText.length >= fullText.length) {
        fullText = finalText;
      }

      let newSessionName = null;
      if (isNewChat) {
        try {
          const titleOptions: GenerateOptions = {
            model: CHAT_MODEL,
            system: "You are a helpful assistant that generates short, concise titles for chat sessions.",
            prompt: `Generate a short, concise title (max 6 words) for a chat based on this initial user message: "${message}". Do not use quotes.`,
            config: { temperature: 0.2, maxOutputTokens: 24 },
          };
          const titleResult: GenerateResponse = await ai.generate(titleOptions);
          const rawTitle = titleResult.text;
          const title = rawTitle.trim().replace(/^['"]+|['"]+$/g, "").trim();
          if (title) {
            await publicChatRef.update({ name: title });
            newSessionName = title;
          }
        } catch (error) {
          console.error("Error generating title:", error);
        }
      }

      const updatedHistory: ChatHistoryMessage[] = [
        ...chatHistory,
        { role: "user", content: message },
        { role: "model", content: fullText },
      ];

      await sessionStore.save(session.id, {
        id: session.id,
        state: sessionData?.state ?? {},
        threads: { [MAIN_THREAD]: toThreadMessages(updatedHistory) },
      });

      const finalCitations = selectFinalCitations(fullText, ragCitations);
      const clientCitations = finalCitations.map((citation) => {
        const { contextText, ...publicCitation } = citation;
        void contextText;
        return publicCitation;
      });
      res.write(
        `data: ${JSON.stringify({
          text: "",
          done: true,
          model: MODEL_NAME,
          sessionId: session.id,
          fullText,
          newSessionName,
          citations: clientCitations,
        })}\n\n`
      );
      res.end();
    } catch (error) {
      sendServerError(res, error);
    }
  }
);
