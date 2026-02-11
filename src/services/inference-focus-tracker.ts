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
  created_at_ms: number;
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
  onPrediction?: (payload: InferencePredictionPayload) => void;
  stream?: MediaStream;
};

const DEFAULT_BASE_URL = 'http://localhost:8001';
const DEFAULT_SAMPLE_INTERVAL_MS = 200; // 5 FPS
const DEFAULT_REQUEST_TIMEOUT_MS = 8000;
const DEFAULT_JPEG_QUALITY = 0.85;

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

export class InferenceFocusTracker {
  private readonly baseUrl: string;
  private readonly sampleIntervalMs: number;
  private readonly requestTimeoutMs: number;
  private readonly jpegQuality: number;

  private stream: MediaStream | null = null;
  private ownsStream = true;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
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
      if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
      }
      const ctx = this.canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get 2D context for frame capture');
      }
      ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      const blob = await canvasToJpegBlob(this.canvas, this.jpegQuality);

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

