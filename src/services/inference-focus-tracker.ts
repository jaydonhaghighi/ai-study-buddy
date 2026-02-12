import type { FaceDetector } from '@mediapipe/tasks-vision';
import { createFaceDetector } from './mediapipe-face-detector';

type FocusPredictionSummary = {
  startTs: number;
  endTs: number;
  focusedMs: number;
  distractedMs: number;
  distractions: number;
  focusPercent: number;
  attentionLabelCounts: Record<string, number>;
  [key: string]: unknown;
};

type SessionStartResponse = {
  session_id: string;
};

type SessionStopResponse = {
  session_id: string;
  summary: FocusPredictionSummary;
};

export type InferencePredictionPayload = {
  session_id: string;
  timestamp_ms: number;
  raw_label: string;
  raw_confidence: number;
  smoothed_label: string;
  smoothed_confidence: number;
  state: string;
  transitioned: boolean;
};

type TrackerOptions = {
  baseUrl?: string;
  sampleIntervalMs?: number;
  requestTimeoutMs?: number;
  jpegQuality?: number;
  cropToFace?: boolean;
  mirrorInput?: boolean;
  facePadding?: number;
  onPrediction?: (payload: InferencePredictionPayload) => void;
  stream?: MediaStream;
};

const DEFAULT_BASE_URL = 'http://localhost:8001';
const DEFAULT_SAMPLE_INTERVAL_MS = 200; // 5 FPS
const DEFAULT_REQUEST_TIMEOUT_MS = 8000;
const DEFAULT_JPEG_QUALITY = 0.85;
const DEFAULT_CROP_TO_FACE = (import.meta.env.VITE_INFERENCE_FACE_CROP ?? '1') !== '0';
const DEFAULT_MIRROR_INPUT = (import.meta.env.VITE_INFERENCE_MIRROR_INPUT ?? '1') !== '0';
const DEFAULT_FACE_PADDING = 0.5;

function getBaseUrl(url?: string): string {
  const configured = (url ?? import.meta.env.VITE_INFERENCE_API_BASE_URL ?? DEFAULT_BASE_URL).trim();
  return configured.replace(/\/+$/, '');
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

function canvasToJpegBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Failed to encode webcam frame to JPEG'));
          return;
        }
        resolve(blob);
      },
      'image/jpeg',
      quality
    );
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export class InferenceFocusTracker {
  private readonly baseUrl: string;
  private readonly sampleIntervalMs: number;
  private readonly requestTimeoutMs: number;
  private readonly jpegQuality: number;
  private readonly cropToFace: boolean;
  private readonly mirrorInput: boolean;
  private readonly facePadding: number;

  private stream: MediaStream | null = null;
  private ownsStream = true;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private detector: FaceDetector | null = null;
  private timer: number | null = null;
  private sessionId: string | null = null;
  private sampleInFlight = false;
  private frameSeq = 0;
  private readonly onPrediction?: (payload: InferencePredictionPayload) => void;
  private lastPrediction: InferencePredictionPayload | null = null;

  constructor(options: TrackerOptions = {}) {
    this.baseUrl = getBaseUrl(options.baseUrl);
    this.sampleIntervalMs = options.sampleIntervalMs ?? DEFAULT_SAMPLE_INTERVAL_MS;
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.jpegQuality = options.jpegQuality ?? DEFAULT_JPEG_QUALITY;
    this.cropToFace = options.cropToFace ?? DEFAULT_CROP_TO_FACE;
    this.mirrorInput = options.mirrorInput ?? DEFAULT_MIRROR_INPUT;
    this.facePadding = options.facePadding ?? DEFAULT_FACE_PADDING;
    this.onPrediction = options.onPrediction;
    if (options.stream) {
      this.stream = options.stream;
      this.ownsStream = false;
    }
  }

  getLastPrediction(): InferencePredictionPayload | null {
    return this.lastPrediction;
  }

  async start(): Promise<void> {
    if (this.timer != null) {
      return;
    }

    const startRes = await fetchWithTimeout(
      `${this.baseUrl}/session/start`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metadata: {
            source: 'web_app',
            userAgent: navigator.userAgent,
          },
        }),
      },
      this.requestTimeoutMs
    );

    if (!startRes.ok) {
      const text = await startRes.text().catch(() => '');
      throw new Error(`Failed to start inference session (${startRes.status}): ${text || 'unknown error'}`);
    }

    const startPayload = (await startRes.json()) as SessionStartResponse;
    this.sessionId = startPayload.session_id;

    try {
      if (!this.stream) {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        this.ownsStream = true;
      }

      this.video = document.createElement('video');
      this.video.playsInline = true;
      this.video.muted = true;
      this.video.srcObject = this.stream;
      await this.video.play();

      this.canvas = document.createElement('canvas');
      if (this.cropToFace) {
        try {
          await this.ensureFaceDetector();
        } catch (error) {
          // Keep streaming even if detector setup fails; full-frame fallback still works.
          console.warn('Inference tracker: face detector init failed, using full-frame input.', error);
          this.detector = null;
        }
      }
      this.timer = window.setInterval(() => {
        void this.sample();
      }, this.sampleIntervalMs);
    } catch (error) {
      await this.cleanupMedia();
      const sid = this.sessionId;
      this.sessionId = null;
      if (sid) {
        void this.stopRemoteSessionQuietly(sid);
      }
      throw error;
    }
  }

  private async sample(): Promise<void> {
    if (this.sampleInFlight) return;
    if (!this.sessionId || !this.video || !this.canvas) return;
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (this.video.videoWidth <= 0 || this.video.videoHeight <= 0) return;

    this.sampleInFlight = true;
    try {
      const blob = await this.captureFrameBlob();

      const formData = new FormData();
      formData.append('file', blob, `frame-${this.frameSeq++}.jpg`);

      const response = await fetchWithTimeout(
        `${this.baseUrl}/session/${this.sessionId}/frame`,
        {
          method: 'POST',
          body: formData,
        },
        this.requestTimeoutMs
      );

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(`Frame upload failed (${response.status}): ${text || 'unknown error'}`);
      }

      // The API returns a prediction payload for this frame; use it for live UI.
      try {
        const payload = (await response.json()) as InferencePredictionPayload;
        if (payload && typeof payload.smoothed_label === 'string') {
          this.lastPrediction = payload;
          this.onPrediction?.(payload);
        }
      } catch {
        // If body can't be parsed (or no body), keep going.
      }
    } catch (error) {
      // Keep tracker alive on transient errors; caller sees warnings in console.
      console.warn('Inference frame upload error:', error);
    } finally {
      this.sampleInFlight = false;
    }
  }

  private async captureFrameBlob(): Promise<Blob> {
    if (!this.video || !this.canvas) {
      throw new Error('Inference tracker media is not initialized');
    }
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D context for frame capture');
    }

    const frameW = this.video.videoWidth;
    const frameH = this.video.videoHeight;
    let sx = 0;
    let sy = 0;
    let sw = frameW;
    let sh = frameH;

    if (this.cropToFace && this.detector) {
      const result = this.detector.detectForVideo(this.video, performance.now());
      const box = result.detections?.[0]?.boundingBox;
      if (box && box.width > 1 && box.height > 1) {
        const px = box.width * this.facePadding;
        const py = box.height * this.facePadding;
        const x1 = clamp(box.originX - px, 0, frameW);
        const y1 = clamp(box.originY - py, 0, frameH);
        const x2 = clamp(box.originX + box.width + px, 0, frameW);
        const y2 = clamp(box.originY + box.height + py, 0, frameH);
        sx = x1;
        sy = y1;
        sw = Math.max(1, x2 - x1);
        sh = Math.max(1, y2 - y1);
      }
    }

    const targetW = Math.max(1, Math.round(sw));
    const targetH = Math.max(1, Math.round(sh));
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, targetW, targetH);
    if (this.mirrorInput) {
      ctx.save();
      ctx.translate(targetW, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(this.video, sx, sy, sw, sh, 0, 0, targetW, targetH);
      ctx.restore();
    } else {
      ctx.drawImage(this.video, sx, sy, sw, sh, 0, 0, targetW, targetH);
    }

    return await canvasToJpegBlob(this.canvas, this.jpegQuality);
  }

  private async ensureFaceDetector(): Promise<void> {
    if (this.detector) {
      return;
    }
    this.detector = await createFaceDetector();
  }

  async stop(): Promise<FocusPredictionSummary> {
    if (this.timer != null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }

    // Let one in-flight request finish before stop.
    const waitStart = Date.now();
    while (this.sampleInFlight && Date.now() - waitStart < this.requestTimeoutMs) {
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }

    await this.cleanupMedia();

    const sid = this.sessionId;
    this.sessionId = null;
    if (!sid) {
      throw new Error('Inference session was not initialized');
    }

    const stopRes = await fetchWithTimeout(
      `${this.baseUrl}/session/${sid}/stop`,
      {
        method: 'POST',
      },
      this.requestTimeoutMs
    );

    if (!stopRes.ok) {
      const text = await stopRes.text().catch(() => '');
      throw new Error(`Failed to stop inference session (${stopRes.status}): ${text || 'unknown error'}`);
    }

    const payload = (await stopRes.json()) as SessionStopResponse;
    return payload.summary;
  }

  private async cleanupMedia(): Promise<void> {
    if (this.stream && this.ownsStream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    this.stream = null;
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
      this.video = null;
    }
    this.canvas = null;
    if (this.detector) {
      this.detector.close();
      this.detector = null;
    }
  }

  private async stopRemoteSessionQuietly(sessionId: string): Promise<void> {
    try {
      await fetchWithTimeout(
        `${this.baseUrl}/session/${sessionId}/stop`,
        {
          method: 'POST',
        },
        this.requestTimeoutMs
      );
    } catch {
      // Ignore cleanup errors.
    }
  }
}

