import { useCallback, useEffect, useRef } from 'react';

type UseFocusAlertAudioParams = {
  soundEnabled: boolean;
  volume: number;
};

type BeepOptions = {
  force?: boolean;
  volumeOverride?: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function useFocusAlertAudio({
  soundEnabled,
  volume,
}: UseFocusAlertAudioParams) {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const settingsRef = useRef({
    soundEnabled,
    volume,
  });

  useEffect(() => {
    settingsRef.current = {
      soundEnabled,
      volume: clamp(volume, 0, 1),
    };
  }, [soundEnabled, volume]);

  useEffect(() => {
    return () => {
      if (audioCtxRef.current) {
        void audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, []);

  const playBeep = useCallback((
    frequencyHz: number,
    durationMs: number,
    gain: number,
    options?: BeepOptions,
  ) => {
    if (typeof window === 'undefined') return;

    const shouldForce = options?.force === true;
    const effectiveVolume = clamp(options?.volumeOverride ?? settingsRef.current.volume, 0, 1);
    if ((!settingsRef.current.soundEnabled && !shouldForce) || effectiveVolume <= 0) return;

    try {
      const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextCtor) return;
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContextCtor();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }

      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();
      oscillator.type = 'sine';
      oscillator.frequency.value = frequencyHz;
      gainNode.gain.value = gain * effectiveVolume;
      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      const now = ctx.currentTime;
      oscillator.start(now);
      oscillator.stop(now + durationMs / 1000);
    } catch {
      // Ignore browser audio failures (autoplay policy or unsupported APIs).
    }
  }, []);

  const playFocusRecoverySound = useCallback(() => {
    playBeep(660, 180, 0.1);
  }, [playBeep]);

  const playDistractedNudgeSound = useCallback(() => {
    playBeep(220, 140, 0.08);
    window.setTimeout(() => {
      playBeep(220, 140, 0.08);
    }, 170);
  }, [playBeep]);

  const playFocusAlertPreviewSound = useCallback((previewVolume: number) => {
    playBeep(220, 140, 0.08, {
      force: true,
      volumeOverride: previewVolume,
    });
    window.setTimeout(() => {
      playBeep(220, 140, 0.08, {
        force: true,
        volumeOverride: previewVolume,
      });
    }, 170);
  }, [playBeep]);

  return {
    playFocusRecoverySound,
    playDistractedNudgeSound,
    playFocusAlertPreviewSound,
  };
}
