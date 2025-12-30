import { useEffect, useMemo, useRef, useState } from 'react';

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

export default function PiCalibrationPreview() {
  const [piUrl, setPiUrl] = useState(getSavedPiUrl());
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [swapRB, setSwapRB] = useState(true);

  const alignedSinceRef = useRef<number | null>(null);
  const [alignedSeconds, setAlignedSeconds] = useState(0);

  const streamUrl = useMemo(() => {
    const base = piUrl.replace(/\/+$/, '');
    // Cache-bust so browser doesn't reuse a dead connection between toggles
    return `${base}/stream.mjpg?ts=${Date.now()}`;
  }, [piUrl, open]);

  const normalizeBase = (raw: string) => raw.trim().replace(/\/+$/, '');

  const stopPreview = async () => {
    const base = normalizeBase(piUrl);
    try {
      await fetch(`${base}/stop`, { method: 'POST' });
    } catch {
      // ignore
    }
    setOpen(false);
    setAlignedSeconds(0);
    alignedSinceRef.current = null;
  };

  const startPreview = async () => {
    const base = normalizeBase(piUrl);
    localStorage.setItem('piPreviewBaseUrl', base);
    setError(null);
    try {
      const res = await fetch(`${base}/start?swap=${swapRB ? '1' : '0'}`, { method: 'POST' });
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
            await stopPreview();
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

  return (
    <div style={{ position: 'fixed', left: 16, bottom: 16, zIndex: 50, width: 320 }}>
      <div style={{ background: 'rgba(20, 20, 20, 0.92)', color: '#fff', borderRadius: 12, padding: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
          <div style={{ fontWeight: 700 }}>Pi Camera Calibration</div>
          {open ? (
            <button onClick={stopPreview} style={{ padding: '6px 10px', cursor: 'pointer' }}>
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
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, fontSize: 12, opacity: 0.9 }}>
            <input
              type="checkbox"
              checked={swapRB}
              onChange={(e) => setSwapRB(e.target.checked)}
              disabled={open}
            />
            Fix colors (swap red/blue)
          </label>
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


