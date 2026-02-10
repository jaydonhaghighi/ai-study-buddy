import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

export type AttentionLabel =
  | 'screen'
  | 'away_left'
  | 'away_right'
  | 'away_up'
  | 'away_down'
  | 'away_unknown';

export type LaptopFocusSummary = {
  startTs: number;
  endTs: number;
  focusedMs: number;
  distractedMs: number;
  distractions: number;
  focusPercent: number;
  attentionLabelCounts: Record<string, number>;
};

type TrackerOptions = {
  sampleIntervalMs?: number;
};

const DEFAULT_SAMPLE_INTERVAL_MS = 400;

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

export class LaptopFocusTracker {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private detector: FaceDetector | null = null;
  private timer: number | null = null;
  private startedAtMs = 0;
  private lastSampleAtMs = 0;
  private focusedMs = 0;
  private distractedMs = 0;
  private distractions = 0;
  private previousFocused: boolean | null = null;
  private readonly sampleIntervalMs: number;
  private readonly labelCounts: Record<AttentionLabel, number> = {
    screen: 0,
    away_left: 0,
    away_right: 0,
    away_up: 0,
    away_down: 0,
    away_unknown: 0,
  };

  constructor(options: TrackerOptions = {}) {
    this.sampleIntervalMs = options.sampleIntervalMs ?? DEFAULT_SAMPLE_INTERVAL_MS;
  }

  async start() {
    if (this.timer != null) {
      return;
    }

    this.startedAtMs = Date.now();
    this.lastSampleAtMs = this.startedAtMs;
    this.focusedMs = 0;
    this.distractedMs = 0;
    this.distractions = 0;
    this.previousFocused = null;
    (Object.keys(this.labelCounts) as AttentionLabel[]).forEach((key) => {
      this.labelCounts[key] = 0;
    });

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      this.video = document.createElement('video');
      this.video.playsInline = true;
      this.video.muted = true;
      this.video.srcObject = this.stream;
      await this.video.play();

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      this.detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
        },
        runningMode: 'VIDEO',
      });

      this.timer = window.setInterval(() => {
        this.sample();
      }, this.sampleIntervalMs);
    } catch (error) {
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      if (this.video) {
        this.video.pause();
        this.video.srcObject = null;
        this.video = null;
      }
      if (this.detector) {
        this.detector.close();
        this.detector = null;
      }
      throw error;
    }
  }

  private sample() {
    if (!this.video || !this.detector) {
      return;
    }
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }

    const nowMs = Date.now();
    const deltaMs = Math.max(0, nowMs - this.lastSampleAtMs);
    this.lastSampleAtMs = nowMs;

    const result = this.detector.detectForVideo(this.video, performance.now());
    const box = result.detections?.[0]?.boundingBox;
    const label = this.classifyAttentionLabel(box);
    const isFocused = label === 'screen';

    this.labelCounts[label] += 1;
    if (isFocused) {
      this.focusedMs += deltaMs;
    } else {
      this.distractedMs += deltaMs;
    }
    if (this.previousFocused === true && !isFocused) {
      this.distractions += 1;
    }
    this.previousFocused = isFocused;
  }

  private classifyAttentionLabel(box: { originX: number; originY: number; width: number; height: number } | undefined): AttentionLabel {
    if (!box || !this.video || this.video.videoWidth <= 0 || this.video.videoHeight <= 0) {
      return 'away_unknown';
    }

    const xCenter = (box.originX + box.width / 2) / this.video.videoWidth;
    const yCenter = (box.originY + box.height / 2) / this.video.videoHeight;
    const x = clamp(xCenter, 0, 1);
    const y = clamp(yCenter, 0, 1);
    const xDelta = x - 0.5;
    const yDelta = y - 0.5;

    // Heuristic: if face stays near screen center, count as focused.
    if (Math.abs(xDelta) < 0.14 && Math.abs(yDelta) < 0.16) {
      return 'screen';
    }

    if (Math.abs(xDelta) >= Math.abs(yDelta)) {
      return xDelta < 0 ? 'away_left' : 'away_right';
    }
    return yDelta < 0 ? 'away_up' : 'away_down';
  }

  async stop(): Promise<LaptopFocusSummary> {
    if (this.timer != null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }

    const endMs = Date.now();
    const totalMeasuredMs = this.focusedMs + this.distractedMs;
    const focusPercent = totalMeasuredMs > 0 ? (this.focusedMs / totalMeasuredMs) * 100 : 0;

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
      this.video = null;
    }
    if (this.detector) {
      this.detector.close();
      this.detector = null;
    }

    const attentionLabelCounts = Object.fromEntries(
      Object.entries(this.labelCounts).filter(([, count]) => count > 0)
    );

    return {
      startTs: Math.floor(this.startedAtMs / 1000),
      endTs: Math.floor(endMs / 1000),
      focusedMs: Math.round(this.focusedMs),
      distractedMs: Math.round(this.distractedMs),
      distractions: this.distractions,
      focusPercent: Number(focusPercent.toFixed(1)),
      attentionLabelCounts,
    };
  }
}

