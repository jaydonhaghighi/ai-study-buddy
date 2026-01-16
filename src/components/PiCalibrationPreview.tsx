import { useEffect, useMemo, useRef, useState } from 'react';
import './PiCalibrationPreview.css';

type Status = {
  enabled: boolean;
  lastFrameTs: number | null;
  faceDetected: boolean;
  aligned: boolean;
  faceBox: number[] | null;
};

function getSavedPiUrl() {
  return localStorage.getItem('piPreviewBaseUrl') || 'http://pi.local:8080';
}

type PiCalibrationPreviewProps = {
  variant?: 'floating' | 'embedded';
  autoStart?: boolean;
  onRequestClose?: () => void;
  onAlignedStable?: () => void;
};

export default function PiCalibrationPreview({
  variant = 'floating',
  autoStart = false,
  onRequestClose,
  onAlignedStable,
}: PiCalibrationPreviewProps) {
  const [piUrl, setPiUrl] = useState(getSavedPiUrl());
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  const alignedSinceRef = useRef<number | null>(null);
  const [alignedSeconds, setAlignedSeconds] = useState(0);
  const autoStartAttemptedRef = useRef(false);

  const streamUrl = useMemo(() => {
    const base = piUrl.replace(/\/+$/, '');
    // Cache-bust so browser doesn't reuse a dead connection between toggles
    return `${base}/stream.mjpg?ts=${Date.now()}`;
  }, [piUrl, open]);

  const normalizeBase = (raw: string) => raw.trim().replace(/\/+$/, '');

  const stopPreview = async (closeModal: boolean = false) => {
    const base = normalizeBase(piUrl);
    try {
      await fetch(`${base}/stop`, { method: 'POST' });
    } catch {
      // ignore
    }
    setOpen(false);
    setAlignedSeconds(0);
    alignedSinceRef.current = null;
    if (closeModal) onRequestClose?.();
  };

  const startPreview = async () => {
    const base = normalizeBase(piUrl);
    localStorage.setItem('piPreviewBaseUrl', base);
    setError(null);
    try {
      const res = await fetch(`${base}/start`, { method: 'POST' });
      const json = await res.json().catch(() => ({}));
      if (!res.ok || json.ok === false) {
        throw new Error(json.error || `Failed to start preview (${res.status})`);
      }
      setOpen(true);
      setAlignedSeconds(0);
      alignedSinceRef.current = null;
    } catch (e: any) {
      setError(e?.message || 'Failed to start preview');
      setOpen(false);
    }
  };

  // Auto-start preview (useful when shown in a modal)
  useEffect(() => {
    if (!autoStart) return;
    if (open) return;
    if (autoStartAttemptedRef.current) return;
    autoStartAttemptedRef.current = true;
    startPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const base = normalizeBase(piUrl);

    const tick = async () => {
      try {
        const res = await fetch(`${base}/status`, { method: 'GET' });
        const json = (await res.json()) as Status;
        if (cancelled) return;
        setStatus(json);

        if (json.aligned) {
          if (alignedSinceRef.current == null) alignedSinceRef.current = Date.now();
          const secs = Math.floor((Date.now() - alignedSinceRef.current) / 1000);
          setAlignedSeconds(secs);
          if (secs >= 3) {
            // Auto-stop after stable alignment
            onAlignedStable?.();
            await stopPreview(false);
          }
        } else {
          alignedSinceRef.current = null;
          setAlignedSeconds(0);
        }
      } catch (e: any) {
        if (cancelled) return;
        setError(e?.message || 'Failed to fetch status');
      }
    };

    const id = window.setInterval(tick, 500);
    tick();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [open, piUrl]);

  if (variant === 'floating') {
    // Keep the old floating widget behavior for dev/demo usage.
    return (
      <div style={{ position: 'fixed', left: 16, bottom: 16, zIndex: 50, width: 320 }}>
        <div style={{ background: 'rgba(20, 20, 20, 0.92)', color: '#fff', borderRadius: 12, padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
            <div style={{ fontWeight: 700 }}>Pi Camera Calibration</div>
            {open ? (
              <button onClick={() => stopPreview(false)} style={{ padding: '6px 10px', cursor: 'pointer' }}>
                Stop
              </button>
            ) : (
              <button onClick={startPreview} style={{ padding: '6px 10px', cursor: 'pointer' }}>
                Calibrate
              </button>
            )}
          </div>

          <div style={{ marginTop: 8 }}>
            <input
              value={piUrl}
              onChange={(e) => setPiUrl(e.target.value)}
              placeholder="http://pi.local:8080"
              style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid rgba(255,255,255,0.25)' }}
              disabled={open}
            />
            <div style={{ fontSize: 12, opacity: 0.8, marginTop: 6 }}>
              Local demo only (HTTP). Video is not saved; stream auto-stops when aligned.
            </div>
          </div>

          {error && <div style={{ color: '#ffb4b4', marginTop: 8, fontSize: 12 }}>{error}</div>}

          {open && (
            <div style={{ marginTop: 10 }}>
              <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.2)' }}>
                <img src={streamUrl} style={{ width: '100%', display: 'block' }} />
              </div>
              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.9 }}>
                <div>Face detected: {status?.faceDetected ? 'yes' : 'no'}</div>
                <div>
                  Aligned: {status?.aligned ? `yes (${alignedSeconds}s)` : 'no'} (auto-stops after 3s)
                </div>
              </div>
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
        <h4 className="pi-calibration-title">Camera calibration</h4>
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

      <input
        value={piUrl}
        onChange={(e) => setPiUrl(e.target.value)}
        placeholder="http://pi.local:8080"
        className="pi-calibration-field"
        disabled={open}
      />
      <p className="pi-calibration-help">
        Local demo only (HTTP). Video is not saved; the preview auto-stops when aligned for ~3 seconds.
      </p>

      {error && <p className="pi-calibration-error">{error}</p>}

      {open && (
        <>
          <div className="pi-calibration-stream">
            <img src={streamUrl} alt="Pi camera preview" />
          </div>
          <div className="pi-calibration-metrics">
            <span className="pi-calibration-metric">
              Face detected: <b>{status?.faceDetected ? 'yes' : 'no'}</b>
            </span>
            <span className="pi-calibration-metric">
              Aligned: <b>{status?.aligned ? `yes (${alignedSeconds}s)` : 'no'}</b>
            </span>
          </div>
        </>
      )}
    </div>
  );
}


