import { useCallback, useEffect, useRef, useState } from 'react';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './PiCalibrationPreview.css';

type Status = {
  enabled: boolean;
  lastFrameTs: number | null;
  faceDetected: boolean;
  aligned: boolean;
  faceBox: [number, number, number, number] | null;
  lastError?: string | null;
};

type PiCalibrationPreviewProps = {
  variant?: 'floating' | 'embedded';
  mode?: 'calibration' | 'monitor';
  autoStart?: boolean;
  /**
   * When set (in seconds), the preview auto-stops once alignment is stable for N seconds.
   * Use `null` to keep the preview running continuously (monitor mode).
   */
  autoStopAfterAlignedSeconds?: number | null;
  /**
   * When alignment becomes stable, call `onAlignedStable`. If `true`, also stop the preview.
   * This is `true` by default for calibration; set to `false` if you want to keep streaming.
   */
  stopOnAlignedStable?: boolean;
  onRequestClose?: () => void;
  onAlignedStable?: () => void;
};

export default function PiCalibrationPreview({
  variant = 'floating',
  mode = 'calibration',
  autoStart = false,
  autoStopAfterAlignedSeconds = mode === 'monitor' ? null : 3,
  stopOnAlignedStable = true,
  onRequestClose,
  onAlignedStable,
}: PiCalibrationPreviewProps) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<FaceDetector | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const alignedSinceRef = useRef<number | null>(null);
  const [alignedSeconds, setAlignedSeconds] = useState(0);
  const autoStartAttemptedRef = useRef(false);
  const alignedStableFiredRef = useRef(false);

  const stopPreview = useCallback(async (closeModal: boolean = false) => {
    if (pollTimerRef.current != null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
    setOpen(false);
    setAlignedSeconds(0);
    alignedSinceRef.current = null;
    alignedStableFiredRef.current = false;
    if (closeModal) onRequestClose?.();
  }, [onRequestClose]);

  const startPreview = useCallback(async () => {
    setError(null);
    try {
      if (!detectorRef.current) {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        detectorRef.current = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
          },
          runningMode: 'VIDEO',
        });
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      streamRef.current = stream;

      setOpen(true);
      await new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => resolve());
      });

      if (!videoRef.current) {
        throw new Error('Video element is not available');
      }
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      setAlignedSeconds(0);
      alignedSinceRef.current = null;
      alignedStableFiredRef.current = false;
    } catch (e: any) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      setError(e?.message || 'Could not access laptop webcam');
      setOpen(false);
    }
  }, []);

  // Auto-start preview (useful when shown in a modal)
  useEffect(() => {
    if (!autoStart) return;
    if (open) return;
    if (autoStartAttemptedRef.current) return;
    autoStartAttemptedRef.current = true;
    startPreview();
  }, [autoStart, open, startPreview]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    const tick = async () => {
      if (!videoRef.current || !detectorRef.current) return;
      if (videoRef.current.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      try {
        const video = videoRef.current;
        const result = detectorRef.current.detectForVideo(video, performance.now());
        const box = result.detections?.[0]?.boundingBox || null;

        const faceDetected = !!box;
        const aligned = !!box && video.videoWidth > 0 && video.videoHeight > 0
          ? (() => {
              const xCenter = (box.originX + box.width / 2) / video.videoWidth;
              const yCenter = (box.originY + box.height / 2) / video.videoHeight;
              const xDelta = Math.abs(xCenter - 0.5);
              const yDelta = Math.abs(yCenter - 0.5);
              return xDelta < 0.14 && yDelta < 0.16;
            })()
          : false;

        const json: Status = {
          enabled: true,
          lastFrameTs: Date.now(),
          faceDetected,
          aligned,
          faceBox: box
            ? [box.originX, box.originY, box.width, box.height]
            : null,
          lastError: null,
        };

        if (cancelled) return;
        setStatus(json);

        if (mode === 'calibration') {
          if (json.aligned) {
            if (alignedSinceRef.current == null) alignedSinceRef.current = Date.now();
            const secs = Math.floor((Date.now() - alignedSinceRef.current) / 1000);
            setAlignedSeconds(secs);
            if (autoStopAfterAlignedSeconds != null && secs >= autoStopAfterAlignedSeconds) {
              if (!alignedStableFiredRef.current) {
                alignedStableFiredRef.current = true;
                if (stopOnAlignedStable) {
                  await stopPreview(false);
                }
                onAlignedStable?.();
              }
            }
          } else {
            alignedSinceRef.current = null;
            setAlignedSeconds(0);
            alignedStableFiredRef.current = false;
          }
        }
      } catch (e: any) {
        if (cancelled) return;
        setStatus({
          enabled: true,
          lastFrameTs: Date.now(),
          faceDetected: false,
          aligned: false,
          faceBox: null,
          lastError: e?.message || 'Failed to read webcam frame',
        });
      }
    };

    const id = window.setInterval(tick, 400);
    pollTimerRef.current = id;
    tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
      if (pollTimerRef.current === id) {
        pollTimerRef.current = null;
      }
    };
  }, [open, mode, autoStopAfterAlignedSeconds, stopOnAlignedStable, onAlignedStable, stopPreview]);

  useEffect(() => {
    return () => {
      if (pollTimerRef.current != null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      detectorRef.current?.close();
      detectorRef.current = null;
    };
  }, []);

  const autoStopLabel =
    autoStopAfterAlignedSeconds == null ? 'continuous' : `auto-stops after ${autoStopAfterAlignedSeconds}s`;

  if (variant === 'floating') {
    // Keep the old floating widget behavior for dev/demo usage.
    return (
      <div style={{ position: 'fixed', right: 336, bottom: 92, zIndex: 50, width: 320 }}>
        <div style={{ background: 'rgba(20, 20, 20, 0.92)', color: '#fff', borderRadius: 12, padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
            <div style={{ fontWeight: 700 }}>{mode === 'monitor' ? 'Laptop camera preview' : 'Laptop camera calibration'}</div>
            {open ? (
              <button onClick={() => stopPreview(false)} style={{ padding: '6px 10px', cursor: 'pointer' }}>
                Stop
              </button>
            ) : (
              <button onClick={startPreview} style={{ padding: '6px 10px', cursor: 'pointer' }}>
                Start
              </button>
            )}
          </div>

          <div style={{ fontSize: 12, opacity: 0.8, marginTop: 6 }}>
            Browser webcam preview. Video is not saved; preview is {autoStopLabel}.
          </div>

          {error && <div style={{ color: '#ffb4b4', marginTop: 8, fontSize: 12 }}>{error}</div>}

          {open && (
            <div style={{ marginTop: 10 }}>
              <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.2)' }}>
                <video ref={videoRef} style={{ width: '100%', display: 'block' }} autoPlay playsInline muted />
              </div>
              {mode === 'calibration' && (
                <div style={{ marginTop: 8, fontSize: 12, opacity: 0.9 }}>
                  <div>Face detected: {status?.faceDetected ? 'yes' : 'no'}</div>
                  <div>
                    Aligned: {status?.aligned ? `yes (${alignedSeconds}s)` : 'no'}
                    {autoStopAfterAlignedSeconds == null ? '' : ` (auto-stops after ${autoStopAfterAlignedSeconds}s)`}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Embedded (modal-friendly) UI
  return (
    <div className="pi-calibration">
      <div className="pi-calibration-header">
        <h4 className="pi-calibration-title">{mode === 'monitor' ? 'Laptop webcam' : 'Laptop webcam calibration'}</h4>
        <div className="pi-calibration-controls">
          {open ? (
            <button className="pi-calibration-btn" onClick={() => stopPreview(true)} type="button">
              Stop
            </button>
          ) : (
            <button className="pi-calibration-btn pi-calibration-btn-primary" onClick={startPreview} type="button">
              Start preview
            </button>
          )}
        </div>
      </div>
    
      {error && <p className="pi-calibration-error">{error}</p>}
      {!error && open && status?.lastError && <p className="pi-calibration-error">{status.lastError}</p>}

      {open && (
        <>
          <div className="pi-calibration-stream">
            <video ref={videoRef} autoPlay playsInline muted />
          </div>

          {mode === 'calibration' && (
            <div className="pi-calibration-metrics">
              <div className="pi-calibration-metric">
                Face: <b>{status?.faceDetected ? 'detected' : 'not detected'}</b>
              </div>
              <div className="pi-calibration-metric">
                Aligned: <b>{status?.aligned ? 'yes' : 'no'}</b>
              </div>
              <div className="pi-calibration-metric">
                Stable: <b>{alignedSeconds}s</b>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}


