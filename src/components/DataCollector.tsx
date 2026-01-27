import { useEffect, useMemo, useRef, useState } from 'react';
import JSZip from 'jszip';
import { FilesetResolver, FaceDetector, FaceLandmarker } from '@mediapipe/tasks-vision';
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
  | { kind: 'calibration'; label: AttentionLabel; awayDirection: string | null; subtitle: string; progressPct: number }
  | { kind: 'countdown'; title: string; subtitle: string; secondsLeft: number; label: AttentionLabel; awayDirection: string | null }
  | { kind: 'capture'; title: string; subtitle: string; label: AttentionLabel; awayDirection: string | null; secondsLeft: number }
  | { kind: 'done' };

function targetText(phase: Phase): string {
  if (phase.kind === 'countdown' || phase.kind === 'capture' || phase.kind === 'calibration') {
    if (phase.label === 'screen') return 'LOOK AT SCREEN';
    const d = (phase.awayDirection || '').toUpperCase();
    if (d) return `LOOK ${d}`;
    return 'LOOK AWAY';
  }
  return '';
}

type Pose = { yaw: number; pitch: number; roll: number };

function radToDeg(r: number) {
  return (r * 180) / Math.PI;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function poseFromMatrix4x4(data: number[] | Float32Array): Pose | null {
  if (!data || data.length < 16) return null;
  // MediaPipe provides a 4x4 matrix. Treat it as column-major (common in WebGL/MP).
  const m = data;
  const r00 = m[0];
  const r10 = m[1];
  const r20 = m[2], r21 = m[6], r22 = m[10];

  // Yaw-Pitch-Roll (Z-Y-X) from rotation matrix
  const pitch = Math.asin(clamp(-r20, -1, 1));
  const yaw = Math.atan2(r10, r00);
  const roll = Math.atan2(r21, r22);
  return { yaw: radToDeg(yaw), pitch: radToDeg(pitch), roll: radToDeg(roll) };
}

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

  const [participant, setParticipant] = useState('');
  const [session, setSession] = useState<'day' | 'night' | ''>('');
  // Fixed collection settings (participants shouldn't change these).
  const placement = 'laptop_webcam';
  const fps = 6;
  const cycles = 2;
  const segmentSeconds = 8;
  // Always-on: we only save frames when a face is detected.
  // Always-on: show an unmirrored preview (and save unmirrored crops).
  const unmirrorPreview = true;
  const [zipProgress, setZipProgress] = useState<{ saved: number; skipped: number } | null>(null);

  const runId = useMemo(() => `run_${Math.floor(Date.now() / 1000)}`, []);
  const participantMode = useMemo(() => {
    if (typeof window === 'undefined') return false;
    const sp = new URLSearchParams(window.location.search);
    return sp.get('mode') === 'participant' || sp.get('participantMode') === '1';
  }, []);

  const [detector, setDetector] = useState<FaceDetector | null>(null);
  const [landmarker, setLandmarker] = useState<FaceLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const calibrationTargetsRef = useRef<Record<AttentionLabel, { yaw: number; pitch: number } | null>>({
    screen: null,
    away_left: null,
    away_right: null,
    away_up: null,
    away_down: null,
  });
  const [poseHud, setPoseHud] = useState<string>('');

  // Participant mode: allow pre-filling via URL params and lock UI down.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const sp = new URLSearchParams(window.location.search);
    const p = sp.get('participant');
    const s = sp.get('session');
    if (p) setParticipant(p);
    if (s === 'day' || s === 'night') setSession(s);
    // placement/fps/cycles/seconds are intentionally fixed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  // Participant mode: try to auto-start the camera on page load.
  useEffect(() => {
    if (!participantMode) return;
    if (running) return;
    if (phase.kind !== 'idle') return;
    if (!deviceId) return;
    // Try once; if the browser blocks it, the manual button still works.
    startCamera().catch((e: any) => {
      setStatus('Click “Start camera” to allow camera access.');
      setError(e?.message || 'Failed to start camera');
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [participantMode, deviceId]);

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
        const lm = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFacialTransformationMatrixes: true,
        } as any);
        if (!cancelled) {
          setDetector(d);
          setLandmarker(lm);
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || 'Failed to initialize face detector');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const detectPose = async (): Promise<Pose | null> => {
    const v = videoRef.current;
    if (!v || !landmarker) return null;
    const now = performance.now();
    const res: any = landmarker.detectForVideo(v, now);
    const mat = res?.facialTransformationMatrixes?.[0]?.data;
    const pose = mat ? poseFromMatrix4x4(mat) : null;
    return pose;
  };

  const poseInRange = (label: AttentionLabel, pose: Pose): boolean => {
    const t = calibrationTargetsRef.current[label];
    if (!t) return true; // if not calibrated yet, don't block
    const yawTol = label === 'screen' ? 10 : 10;
    const pitchTol = label === 'screen' ? 10 : 10;
    return Math.abs(pose.yaw - t.yaw) <= yawTol && Math.abs(pose.pitch - t.pitch) <= pitchTol;
  };

  const calibrationSubtitleFor = (label: AttentionLabel) => {
    switch (label) {
      case 'screen': return 'Look at the center of your screen. Hold still.';
      case 'away_left': return 'Turn your head LEFT (nose points to left edge of screen). Hold still.';
      case 'away_right': return 'Turn your head RIGHT (nose points to right edge of screen). Hold still.';
      case 'away_up': return 'Tilt head UP (look above the screen). Hold still.';
      case 'away_down': return 'Tilt head DOWN (look toward keyboard/desk). Hold still.';
    }
  };

  const runCalibration = async () => {
    const HOLD_SECONDS = 3;
    const steps: { label: AttentionLabel; awayDirection: string | null }[] = [
      { label: 'screen', awayDirection: null },
      { label: 'away_left', awayDirection: 'left' },
      { label: 'away_right', awayDirection: 'right' },
      { label: 'away_up', awayDirection: 'up' },
      { label: 'away_down', awayDirection: 'down' },
    ];

    setStatus('Calibration: follow the prompts…');
    for (const step of steps) {
      // Give a moment to move into position
      for (let s = 2; s >= 1; s--) {
        setPhase({
          kind: 'calibration',
          label: step.label,
          awayDirection: step.awayDirection,
          subtitle: calibrationSubtitleFor(step.label),
          progressPct: 0,
        });
        beep();
        // eslint-disable-next-line no-await-in-loop
        await sleep(800);
      }

      const samples: Pose[] = [];
      let stableSince: number | null = null;
      while (true) {
        if (cancelRef.current) throw new Error('__SB_CANCELLED__');
        // eslint-disable-next-line no-await-in-loop
        const pose = await detectPose();
        if (!pose) {
          stableSince = null;
          setPoseHud('No face landmarks (move into frame)');
          setPhase({
            kind: 'calibration',
            label: step.label,
            awayDirection: step.awayDirection,
            subtitle: calibrationSubtitleFor(step.label),
            progressPct: 0,
          });
          // eslint-disable-next-line no-await-in-loop
          await sleep(120);
          continue;
        }

        samples.push(pose);
        while (samples.length > 18) samples.shift();

        // compute simple stability (stddev) once we have enough samples
        let okStable = false;
        if (samples.length >= 12) {
          const meanYaw = samples.reduce((a, p) => a + p.yaw, 0) / samples.length;
          const meanPitch = samples.reduce((a, p) => a + p.pitch, 0) / samples.length;
          const varYaw = samples.reduce((a, p) => a + (p.yaw - meanYaw) ** 2, 0) / samples.length;
          const varPitch = samples.reduce((a, p) => a + (p.pitch - meanPitch) ** 2, 0) / samples.length;
          const stdYaw = Math.sqrt(varYaw);
          const stdPitch = Math.sqrt(varPitch);
          okStable = stdYaw < 2.8 && stdPitch < 2.8;
        }

        if (okStable) {
          if (stableSince == null) stableSince = Date.now();
        } else {
          stableSince = null;
        }

        const progress = stableSince
          ? clamp(((Date.now() - stableSince) / 1000) / HOLD_SECONDS, 0, 1)
          : 0;
        setPoseHud(`Yaw ${pose.yaw.toFixed(0)}° • Pitch ${pose.pitch.toFixed(0)}° • ${okStable ? 'Hold…' : 'Adjust…'}`);
        setPhase({
          kind: 'calibration',
          label: step.label,
          awayDirection: step.awayDirection,
          subtitle: calibrationSubtitleFor(step.label),
          progressPct: Math.round(progress * 100),
        });

        if (progress >= 1) {
          const meanYaw = samples.reduce((a, p) => a + p.yaw, 0) / samples.length;
          const meanPitch = samples.reduce((a, p) => a + p.pitch, 0) / samples.length;
          calibrationTargetsRef.current[step.label] = { yaw: meanYaw, pitch: meanPitch };
          beep();
          break;
        }

        // eslint-disable-next-line no-await-in-loop
        await sleep(120);
      }
    }
    setStatus('Calibration complete.');
  };

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

    // Larger padding keeps more forehead/chin/cheeks for extreme head poses.
    const pad = 0.5;
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
    if (!participant.trim()) {
      setError('Please enter a participant id (e.g. p01).');
      return;
    }
    if (session !== 'day' && session !== 'night') {
      setError('Please select session: day or night.');
      return;
    }
    if (!detector || !landmarker) {
      setError('Vision models not ready yet. Please wait a second and try again.');
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
      // Enforce consistent head-turn standards (per-person calibration).
      await runCalibration();
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
            setPhase({
              kind: 'countdown',
              title: 'LOOK AT SCREEN',
              subtitle: LOOKING_INSTRUCTION,
              secondsLeft: s,
              label: 'screen',
              awayDirection: null,
            });
            beep();
            // eslint-disable-next-line no-await-in-loop
            await sleep(1000);
          }
          beep();

          // Capture LOOKING
          let lookStart: number | null = null;
          while (true) {
            ensureNotCancelled();
            const { box } = await detectFace();
            drawOverlay(box);

            const pose = await detectPose();
            if (pose) {
              setPoseHud(`Yaw ${pose.yaw.toFixed(0)}° • Pitch ${pose.pitch.toFixed(0)}° • ${poseInRange('screen', pose) ? 'OK' : 'Adjust…'}`);
            }

            const aligned = !!box && !!pose && poseInRange('screen', pose);
            if (!lookStart) {
              setPhase({
                kind: 'capture',
                title: 'LOOK AT SCREEN',
                subtitle: `${LOOKING_INSTRUCTION} (Align to start)`,
                label: 'screen',
                awayDirection: null,
                secondsLeft: segmentSeconds,
              });
              if (aligned) {
                lookStart = Date.now();
                beep();
              }
              // eslint-disable-next-line no-await-in-loop
              await sleep(1000 / Math.max(1, fps));
              continue;
            }

            const lookEnd = lookStart + segmentSeconds * 1000;
            if (Date.now() >= lookEnd) break;

            if (aligned) {
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
            } else {
              setZipProgress((p) => (p ? { ...p, skipped: p.skipped + 1 } : p));
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
              setPhase({
                kind: 'countdown',
                title: 'LOOK AWAY',
                subtitle: away.text,
                secondsLeft: s,
                label: away.label,
                awayDirection: away.dir,
              });
              beep();
              // eslint-disable-next-line no-await-in-loop
              await sleep(1000);
            }
            beep();

            let awayStart: number | null = null;
            while (true) {
              ensureNotCancelled();
              const { box } = await detectFace();
              drawOverlay(box);

              const pose = await detectPose();
              if (pose) {
                setPoseHud(
                  `Yaw ${pose.yaw.toFixed(0)}° • Pitch ${pose.pitch.toFixed(0)}° • ${poseInRange(away.label, pose) ? 'OK' : 'Adjust…'}`
                );
              }

              const aligned = !!box && !!pose && poseInRange(away.label, pose);
              if (!awayStart) {
                setPhase({
                  kind: 'capture',
                  title: 'LOOK AWAY',
                  subtitle: `${away.text} (Align to start)`,
                  label: away.label,
                  awayDirection: away.dir,
                  secondsLeft: segmentSeconds,
                });
                if (aligned) {
                  awayStart = Date.now();
                  beep();
                }
                // eslint-disable-next-line no-await-in-loop
                await sleep(1000 / Math.max(1, fps));
                continue;
              }

              const awayEnd = awayStart + segmentSeconds * 1000;
              if (Date.now() >= awayEnd) break;

              if (aligned) {
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
              } else {
                setZipProgress((p) => (p ? { ...p, skipped: p.skipped + 1 } : p));
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

  return (
    <div className="dc">
      <div className="dc-header">
        <div>
          <div className="dc-title">Data collection</div>
        </div>
      </div>

      <div className="dc-grid">
        <div className="dc-left">
          <div className="dc-card">
            <div className="dc-card-title">Setup</div>
            <div className="dc-form">
              <label>
                Participant&apos;s Name
                <input
                  value={participant}
                  onChange={(e) => setParticipant(e.target.value)}
                  disabled={running || participantMode}
                  placeholder="e.g. John Doe"
                />
              </label>
              <label>
                Session
                <select
                  value={session}
                  onChange={(e) => setSession((e.target.value as any) || '')}
                  disabled={running || participantMode}
                >
                  <option value="">Select…</option>
                  <option value="day">day</option>
                  <option value="night">night</option>
                </select>
              </label>
              <label>
                Camera
                <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)} disabled={running || participantMode}>
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
                  Start Calibration
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

          {error && <div className="dc-error">{error}</div>}
          {conditionUi()}
        </div>

        <div className="dc-right">
          <div className="dc-preview dc-preview-unmirror">
            <video ref={videoRef} className="dc-video" playsInline muted />
            <canvas ref={overlayRef} className="dc-overlay" />
            {(phase.kind === 'countdown' || phase.kind === 'capture' || phase.kind === 'calibration') && (
              <div className="dc-videoPrompt">
                <div className="dc-videoPromptMain">{targetText(phase)}</div>
                <div className="dc-videoPromptSub">
                  {phase.kind === 'countdown'
                    ? `Starting in ${phase.secondsLeft}s`
                    : phase.kind === 'capture'
                      ? `${phase.secondsLeft}s left`
                      : `Calibration ${phase.progressPct}% • ${poseHud || 'Hold still…'}`}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <canvas ref={faceCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}

