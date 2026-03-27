import {
  Timestamp,
  doc,
  onSnapshot,
  serverTimestamp,
  setDoc,
} from 'firebase/firestore';
import { db } from '../firebase-config';
import type { FocusAlertSettings } from '../types';

export const MIN_FOCUS_ALERT_DELAY_MINUTES = 1;
export const MAX_FOCUS_ALERT_DELAY_MINUTES = 15;
const FOCUS_ALERT_SETTINGS_STORAGE_KEY_PREFIX = 'studybuddy.focusAlerts';

export const DEFAULT_FOCUS_ALERT_SETTINGS: FocusAlertSettings = {
  nudgeDelayMinutes: 1,
  soundEnabled: true,
  volume: 0.6,
  updatedAt: null,
};

type FocusAlertSettingsInput = Pick<FocusAlertSettings, 'nudgeDelayMinutes' | 'soundEnabled' | 'volume'>;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toDate(value: unknown): Date | null {
  if (value instanceof Timestamp) return value.toDate();
  if (value instanceof Date) return value;
  if (typeof value === 'string') {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  return null;
}

function storageKey(userId: string): string {
  return `${FOCUS_ALERT_SETTINGS_STORAGE_KEY_PREFIX}.${userId}`;
}

function canUseLocalStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

export function normalizeFocusAlertDelayMinutes(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_FOCUS_ALERT_SETTINGS.nudgeDelayMinutes;
  }
  return clamp(Math.round(value), MIN_FOCUS_ALERT_DELAY_MINUTES, MAX_FOCUS_ALERT_DELAY_MINUTES);
}

export function normalizeFocusAlertVolume(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_FOCUS_ALERT_SETTINGS.volume;
  }
  return Number(clamp(value, 0, 1).toFixed(2));
}

function normalizeFocusAlertSettings(input: FocusAlertSettingsInput): FocusAlertSettingsInput {
  return {
    nudgeDelayMinutes: normalizeFocusAlertDelayMinutes(input.nudgeDelayMinutes),
    soundEnabled: typeof input.soundEnabled === 'boolean'
      ? input.soundEnabled
      : DEFAULT_FOCUS_ALERT_SETTINGS.soundEnabled,
    volume: normalizeFocusAlertVolume(input.volume),
  };
}

function buildFocusAlertSettings(
  input: Partial<FocusAlertSettingsInput> | null | undefined,
  updatedAt: unknown,
): FocusAlertSettings {
  return {
    nudgeDelayMinutes: normalizeFocusAlertDelayMinutes(input?.nudgeDelayMinutes),
    soundEnabled: typeof input?.soundEnabled === 'boolean'
      ? input.soundEnabled
      : DEFAULT_FOCUS_ALERT_SETTINGS.soundEnabled,
    volume: normalizeFocusAlertVolume(input?.volume),
    updatedAt: toDate(updatedAt),
  };
}

export function parseFocusAlertSettings(data: Record<string, unknown> | null | undefined): FocusAlertSettings {
  const focusAlerts =
    data && typeof data.focusAlerts === 'object' && data.focusAlerts !== null
      ? (data.focusAlerts as Record<string, unknown>)
      : null;

  return buildFocusAlertSettings(focusAlerts, data?.updatedAt);
}

export function readLocalFocusAlertSettings(userId: string): FocusAlertSettings {
  if (!canUseLocalStorage()) return DEFAULT_FOCUS_ALERT_SETTINGS;

  try {
    const raw = window.localStorage.getItem(storageKey(userId));
    if (!raw) return DEFAULT_FOCUS_ALERT_SETTINGS;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return buildFocusAlertSettings(parsed, parsed.updatedAt);
  } catch {
    return DEFAULT_FOCUS_ALERT_SETTINGS;
  }
}

export function saveLocalFocusAlertSettings(
  userId: string,
  settings: FocusAlertSettingsInput,
): FocusAlertSettings {
  const normalized = normalizeFocusAlertSettings(settings);
  const next: FocusAlertSettings = {
    ...normalized,
    updatedAt: new Date(),
  };

  if (!canUseLocalStorage()) return next;

  try {
    window.localStorage.setItem(
      storageKey(userId),
      JSON.stringify({
        ...normalized,
        updatedAt: next.updatedAt?.toISOString() ?? null,
      }),
    );
  } catch {
    // Best-effort local fallback only.
  }

  return next;
}

export function subscribeToFocusAlertSettings(
  userId: string,
  onValue: (settings: FocusAlertSettings) => void,
  onError?: (error: Error) => void,
) {
  const ref = doc(db, 'userSettings', userId);
  return onSnapshot(
    ref,
    (snapshot) => {
      onValue(parseFocusAlertSettings(snapshot.exists() ? snapshot.data() : null));
    },
    (error) => {
      onError?.(error);
    },
  );
}

export async function saveFocusAlertSettings(
  userId: string,
  settings: FocusAlertSettingsInput,
): Promise<void> {
  const normalized = normalizeFocusAlertSettings(settings);
  await setDoc(
    doc(db, 'userSettings', userId),
    {
      userId,
      focusAlerts: normalized,
      updatedAt: serverTimestamp(),
    },
    { merge: true },
  );
}
