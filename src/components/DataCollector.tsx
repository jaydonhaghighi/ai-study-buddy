import { useEffect, useMemo, useRef, useState } from 'react';
import JSZip from 'jszip';
import { FilesetResolver, FaceDetector } from '@mediapipe/tasks-vision';
import './DataCollector.css';

type Condition = {
  name: string;
  instructions: string;
  skippable?: boolean;
};

type AttentionLabel = 'screen' | 'away_left' | 'away_right' | 'away_up' | 'away_down';

const CONDITIONS: Condition[] = [
  { name: 'normal', instructions: 'Normal posture, normal lighting. Sit as you would during study.' },
  { name: 'lean_back', instructions: 'Lean back slightly (relaxed posture).' },
  { name: 'lean_forward', instructions: 'Lean forward slightly (concentrating posture).' },
  { name: 'glasses', instructions: 'If you have glasses: put them on. If not, you can skip this condition.', skippable: true },
  { name: 'dim_light', instructions: 'Dim the room a bit if possible. If not possible, you can skip.', skippable: true },
];

const AWAY_TARGETS = [
  { dir: 'left', label: 'away_left' as const, text: 'Look OFF-SCREEN to your LEFT (turn head slightly).' },
  { dir: 'right', label: 'away_right' as const, text: 'Look OFF-SCREEN to your RIGHT (turn head slightly).' },
  { dir: 'up', label: 'away_up' as const, text: 'Look OFF-SCREEN UP (above your screen).' },
  { dir: 'down', label: 'away_down' as const, text: 'Look DOWN (keyboard/desk).' },
];

const LOOKING_INSTRUCTION =
  'Look at the CENTER of your screen (as if reading). Keep your face in frame.';

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function slug(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function beep() {
  try {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'sine';
    o.frequency.value = 880;
    g.gain.value = 0.04;
    o.connect(g);
    g.connect(ctx.destination);
    o.start();
    setTimeout(() => {
      o.stop();
      ctx.close().catch(() => {});
    }, 120);
  } catch {
    // ignore
  }
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'ready' }
  | { kind: 'condition'; conditionIdx: number }
  | { kind: 'countdown'; title: string; subtitle: string; secondsLeft: number }
  | { kind: 'capture'; title: string; subtitle: string; label: AttentionLabel; awayDirection: string | null; secondsLeft: number }
  | { kind: 'done' };

export default function DataCollector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const cancelRef = useRef(false);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const [participant, setParticipant] = useState('p01');
  const [session, setSession] = useState('day');
  const [placement, setPlacement] = useState('laptop_webcam');
  const [fps, setFps] = useState(6);
  const [cycles, setCycles] = useState(6);
  const [lookSeconds, setLookSeconds] = useState(6);
  const [awaySeconds, setAwaySeconds] = useState(6);
  // Always-on: we only save frames when a face is detected.
  const requireFace = true;
  // Always-on: show an unmirrored preview (and save unmirrored crops).
  const unmirrorPreview = true;
  const [zipProgress, setZipProgress] = useState<{ saved: number; skipped: number } | null>(null);

  const runId = useMemo(() => `run_${Math.floor(Date.now() / 1000)}`, []);

  const [detector, setDetector] = useState<FaceDetector | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    (async () => {
      try {
        // Prompt once so enumerateDevices shows labels.
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        tmp.getTracks().forEach((t) => t.stop());
      } catch {
        // ignore; we’ll show errors when starting.
      }
      const ds = await navigator.mediaDevices.enumerateDevices();
      const cams = ds.filter((d) => d.kind === 'videoinput');
      setDevices(cams);
      if (!deviceId && cams[0]?.deviceId) setDeviceId(cams[0].deviceId);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        const d = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
          },
          runningMode: 'VIDEO',
        });
        if (!cancelled) setDetector(d);
      } catch (e: any) {
        if (!cancelled) setError(e?.message || 'Failed to initialize face detector');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const startCamera = async () => {
    setError(null);
    setStatus('Requesting camera permission…');
    const constraints: MediaStreamConstraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : true,
      audio: false,
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    streamRef.current = stream;
    const v = videoRef.current!;
    v.srcObject = stream;
    await v.play();
    setStatus('Camera started.');
    setPhase({ kind: 'ready' });
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStatus('Camera stopped.');
    setPhase({ kind: 'idle' });
  };

  const drawOverlay = (box: { x: number; y: number; w: number; h: number } | null) => {
    const v = videoRef.current;
    const c = overlayRef.current;
    if (!v || !c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
    ctx.clearRect(0, 0, c.width, c.height);
    if (box) {
      ctx.strokeStyle = '#00aa00';
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.w, box.h);
    }
  };

  const detectFace = async (): Promise<{ box: { x: number; y: number; w: number; h: number } | null }> => {
    const v = videoRef.current;
    if (!v || !detector) return { box: null };
    const now = performance.now();
    const res = detector.detectForVideo(v, now);
    const det = res.detections?.[0];
    const bb = det?.boundingBox;
    if (!bb) return { box: null };
    return {
      box: { x: bb.originX, y: bb.originY, w: bb.width, h: bb.height },
    };
  };

  const cropFaceToJpeg = async (box: { x: number; y: number; w: number; h: number }, size = 224): Promise<Blob> => {
    const v = videoRef.current!;
    const faceCanvas = faceCanvasRef.current!;
    const ctx = faceCanvas.getContext('2d')!;

    const pad = 0.3;
    const px = box.w * pad;
    const py = box.h * pad;
    const x1 = Math.max(0, box.x - px);
    const y1 = Math.max(0, box.y - py);
    const x2 = Math.min(v.videoWidth, box.x + box.w + px);
    const y2 = Math.min(v.videoHeight, box.y + box.h + py);

    faceCanvas.width = size;
    faceCanvas.height = size;
    ctx.clearRect(0, 0, size, size);
    if (unmirrorPreview) {
      // Flip horizontally so the saved crops match the unmirrored preview.
      ctx.save();
      ctx.translate(size, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(v, x1, y1, x2 - x1, y2 - y1, 0, 0, size, size);
      ctx.restore();
    } else {
      ctx.drawImage(v, x1, y1, x2 - x1, y2 - y1, 0, 0, size, size);
    }

    return await new Promise((resolve) => faceCanvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.9));
  };

  const runWizard = async () => {
    if (!detector) {
      setError('Face detector not ready yet. Please wait a second and try again.');
      return;
    }
    const v = videoRef.current;
    if (!v || !v.videoWidth) {
      setError('Camera not started or no frames yet.');
      return;
    }

    setRunning(true);
    cancelRef.current = false;
    setError(null);
    setZipProgress({ saved: 0, skipped: 0 });
    const zip = new JSZip();
    const metaLines: string[] = [];

    try {
      const ensureNotCancelled = () => {
        if (cancelRef.current) throw new Error('__SB_CANCELLED__');
      };
      for (let ci = 0; ci < CONDITIONS.length; ci++) {
        ensureNotCancelled();
        setPhase({ kind: 'condition', conditionIdx: ci });
        // Wait for user action via UI buttons
        // eslint-disable-next-line no-await-in-loop
        const action = await new Promise<'start' | 'skip'>((resolve) => {
          (window as any).__sbResolveCondition = resolve;
        });
        (window as any).__sbResolveCondition = null;
        if (action === 'skip') continue;

        const cond = CONDITIONS[ci];
        const conditionTag = `${placement}_${slug(cond.name)}`;

        // A "cycle" is a full set: screen + (left, right, up, down).
        for (let setIdx = 0; setIdx < cycles; setIdx++) {
          ensureNotCancelled();
          // LOOKING countdown
          for (let s = 3; s >= 1; s--) {
            ensureNotCancelled();
            setPhase({ kind: 'countdown', title: 'LOOK AT SCREEN', subtitle: LOOKING_INSTRUCTION, secondsLeft: s });
            beep();
            // eslint-disable-next-line no-await-in-loop
            await sleep(1000);
          }
          beep();

          // Capture LOOKING
          const lookEnd = Date.now() + lookSeconds * 1000;
          while (Date.now() < lookEnd) {
            ensureNotCancelled();
            const { box } = await detectFace();
            drawOverlay(box);
            if (!box) {
              setZipProgress((p) => (p ? { ...p, skipped: p.skipped + 1 } : p));
              if (requireFace) {
                // eslint-disable-next-line no-await-in-loop
                await sleep(1000 / Math.max(1, fps));
                continue;
              }
            }

            if (box) {
              const blob = await cropFaceToJpeg(box, 224);
              const ts = Date.now();
              const fname = `${ts}.jpg`;
              const path = `${runId}/face/${participant}/${session}/${conditionTag}/screen/${fname}`;
              zip.file(path, blob);
              metaLines.push(
                JSON.stringify({
                  label: 'screen',
                  timestamp: ts / 1000,
                  participant,
                  session,
                  basePlacement: placement,
                  condition: conditionTag,
                  instruction: LOOKING_INSTRUCTION,
                  awayDirection: null,
                  faceBox: box,
                  facePath: path,
                })
              );
              setZipProgress((p) => (p ? { ...p, saved: p.saved + 1 } : p));
            }
            const secsLeft = Math.max(0, Math.ceil((lookEnd - Date.now()) / 1000));
            setPhase({ kind: 'capture', title: 'LOOK AT SCREEN', subtitle: LOOKING_INSTRUCTION, label: 'screen', awayDirection: null, secondsLeft: secsLeft });
            // eslint-disable-next-line no-await-in-loop
            await sleep(1000 / Math.max(1, fps));
          }

          // Now capture each away direction once in this cycle
          for (let aIdx = 0; aIdx < AWAY_TARGETS.length; aIdx++) {
            ensureNotCancelled();
            const away = AWAY_TARGETS[aIdx];
            for (let s = 3; s >= 1; s--) {
              ensureNotCancelled();
              setPhase({ kind: 'countdown', title: 'LOOK AWAY', subtitle: away.text, secondsLeft: s });
              beep();
              // eslint-disable-next-line no-await-in-loop
              await sleep(1000);
            }
            beep();

            const awayEnd = Date.now() + awaySeconds * 1000;
            while (Date.now() < awayEnd) {
              ensureNotCancelled();
              const { box } = await detectFace();
              drawOverlay(box);
              if (!box) {
                setZipProgress((p) => (p ? { ...p, skipped: p.skipped + 1 } : p));
                if (requireFace) {
                  // eslint-disable-next-line no-await-in-loop
                  await sleep(1000 / Math.max(1, fps));
                  continue;
                }
              }

              if (box) {
                const blob = await cropFaceToJpeg(box, 224);
                const ts = Date.now();
                const fname = `${ts}.jpg`;
                const path = `${runId}/face/${participant}/${session}/${conditionTag}/${away.label}/${fname}`;
                zip.file(path, blob);
                metaLines.push(
                  JSON.stringify({
                    label: away.label,
                    timestamp: ts / 1000,
                    participant,
                    session,
                    basePlacement: placement,
                    condition: conditionTag,
                    instruction: away.text,
                    awayDirection: away.dir,
                    faceBox: box,
                    facePath: path,
                  })
                );
                setZipProgress((p) => (p ? { ...p, saved: p.saved + 1 } : p));
              }
              const secsLeft = Math.max(0, Math.ceil((awayEnd - Date.now()) / 1000));
              setPhase({ kind: 'capture', title: 'LOOK AWAY', subtitle: away.text, label: away.label, awayDirection: away.dir, secondsLeft: secsLeft });
              // eslint-disable-next-line no-await-in-loop
              await sleep(1000 / Math.max(1, fps));
            }
          }
        }
      }

      zip.file(`${runId}/meta.jsonl`, metaLines.join('\n') + '\n');
      setPhase({ kind: 'done' });
      setStatus('Packaging zip…');
      const blob = await zip.generateAsync({ type: 'blob' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${runId}.zip`;
      a.click();
      setStatus(`Downloaded ${runId}.zip`);
    } catch (e: any) {
      const msg = e?.message || 'Collection failed';
      if (msg === '__SB_CANCELLED__') {
        // Download what we have so far.
        zip.file(`${runId}/meta.jsonl`, metaLines.join('\n') + '\n');
        setStatus('Stopping… packaging partial zip…');
        const blob = await zip.generateAsync({ type: 'blob' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${runId}_partial.zip`;
        a.click();
        setStatus(`Stopped. Downloaded ${runId}_partial.zip`);
        setPhase({ kind: 'ready' });
      } else {
        setError(msg);
      }
    } finally {
      setRunning(false);
    }
  };

  const conditionUi = () => {
    if (phase.kind !== 'condition') return null;
    const cond = CONDITIONS[phase.conditionIdx];
    return (
      <div className="dc-card">
        <div className="dc-card-title">Condition: {cond.name}</div>
        <div className="dc-card-sub">{cond.instructions}</div>
        <div className="dc-row">
          <button
            className="dc-btn dc-btn-primary"
            onClick={() => (window as any).__sbResolveCondition?.('start')}
            type="button"
          >
            Start condition
          </button>
          {cond.skippable && (
            <button className="dc-btn" onClick={() => (window as any).__sbResolveCondition?.('skip')} type="button">
              Skip
            </button>
          )}
        </div>
      </div>
    );
  };

  const overlayText = () => {
    if (phase.kind === 'countdown') return `${phase.title} — starting in ${phase.secondsLeft}s\n${phase.subtitle}`;
    if (phase.kind === 'capture') return `${phase.title} — ${phase.secondsLeft}s left\n${phase.subtitle}`;
    if (phase.kind === 'ready') return 'Ready. Click “Start guided session”.';
    if (phase.kind === 'done') return 'Done. Download should have started.';
    if (phase.kind === 'idle') return 'Start the camera to begin.';
    return '';
  };

  return (
    <div className="dc">
      <div className="dc-header">
        <div>
          <div className="dc-title">Data collection (looking vs not-looking)</div>
          <div className="dc-sub">Runs fully in the browser. Downloads a zip you can send back.</div>
        </div>
      </div>

      {error && <div className="dc-error">{error}</div>}
      <div className="dc-grid">
        <div className="dc-left">
          <div className="dc-card">
            <div className="dc-card-title">Setup</div>
            <div className="dc-form">
              <label>
                Participant
                <input value={participant} onChange={(e) => setParticipant(e.target.value)} disabled={running} />
              </label>
              <label>
                Session
                <input value={session} onChange={(e) => setSession(e.target.value)} disabled={running} />
              </label>
              <label>
                Placement
                <input value={placement} onChange={(e) => setPlacement(e.target.value)} disabled={running} />
              </label>
              <div className="dc-row">
                <label style={{ flex: 1 }}>
                  FPS
                  <input
                    type="number"
                    min={1}
                    max={15}
                    value={fps}
                    onChange={(e) => setFps(Math.max(1, Number(e.target.value || 1)))}
                    disabled={running}
                  />
                </label>
                <label style={{ flex: 1 }}>
                  Cycles (screen + 4 away)
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={cycles}
                    onChange={(e) => setCycles(Math.max(1, Number(e.target.value || 1)))}
                    disabled={running}
                  />
                </label>
              </div>
              <div className="dc-row">
                <label style={{ flex: 1 }}>
                  Look seconds
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={lookSeconds}
                    onChange={(e) => setLookSeconds(Math.max(1, Number(e.target.value || 1)))}
                    disabled={running}
                  />
                </label>
                <label style={{ flex: 1 }}>
                  Away seconds
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={awaySeconds}
                    onChange={(e) => setAwaySeconds(Math.max(1, Number(e.target.value || 1)))}
                    disabled={running}
                  />
                </label>
              </div>
              <label>
                Camera
                <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)} disabled={running}>
                  {devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `Camera ${d.deviceId.slice(0, 6)}…`}
                    </option>
                  ))}
                </select>
              </label>

              <div className="dc-row">
                {phase.kind === 'idle' ? (
                  <button className="dc-btn dc-btn-primary" onClick={startCamera} type="button">
                    Start camera
                  </button>
                ) : (
                  <button className="dc-btn" onClick={stopCamera} type="button" disabled={running}>
                    Stop camera
                  </button>
                )}
                <button
                  className="dc-btn dc-btn-primary"
                  onClick={runWizard}
                  disabled={running || phase.kind === 'idle' || !detector}
                  type="button"
                >
                  Start guided session
                </button>
                <button
                  className="dc-btn dc-btn-danger"
                  onClick={() => {
                    cancelRef.current = true;
                    setStatus('Stopping…');
                  }}
                  disabled={!running}
                  type="button"
                >
                  Stop
                </button>
              </div>
            </div>
            <div className="dc-meta">
              <div>Status: {status || '—'}</div>
              <div>Detector: {detector ? 'ready' : 'loading…'}</div>
              <div>Run id: {runId}</div>
              {zipProgress && (
                <div>
                  Saved: {zipProgress.saved} • Skipped(no face): {zipProgress.skipped}
                </div>
              )}
            </div>
          </div>

          {conditionUi()}
        </div>

        <div className="dc-right">
          <div className="dc-preview dc-preview-unmirror">
            <video ref={videoRef} className="dc-video" playsInline muted />
            <canvas ref={overlayRef} className="dc-overlay" />
            <div className="dc-prompt">{overlayText()}</div>
          </div>
        </div>
      </div>

      <canvas ref={faceCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}

