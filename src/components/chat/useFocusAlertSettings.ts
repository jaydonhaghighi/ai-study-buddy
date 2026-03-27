import { useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import type { FocusAlertSettings } from '../../types';
import {
  DEFAULT_FOCUS_ALERT_SETTINGS,
  readLocalFocusAlertSettings,
  saveLocalFocusAlertSettings,
  saveFocusAlertSettings,
  subscribeToFocusAlertSettings,
} from '../../services/focus-alert-settings-service';

type ToastVariant = 'success' | 'warning' | 'info';

type UseFocusAlertSettingsParams = {
  user: User | null;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function isPermissionDenied(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { code?: string }).code === 'permission-denied';
}

export function useFocusAlertSettings({
  user,
  showToast,
}: UseFocusAlertSettingsParams) {
  const [focusAlertSettings, setFocusAlertSettings] = useState<FocusAlertSettings>(DEFAULT_FOCUS_ALERT_SETTINGS);
  const [focusAlertSettingsLoading, setFocusAlertSettingsLoading] = useState(false);
  const [focusAlertSettingsSaving, setFocusAlertSettingsSaving] = useState(false);
  const shownLocalFallbackToastRef = useRef(false);

  useEffect(() => {
    if (!user) {
      setFocusAlertSettings(DEFAULT_FOCUS_ALERT_SETTINGS);
      setFocusAlertSettingsLoading(false);
      setFocusAlertSettingsSaving(false);
      shownLocalFallbackToastRef.current = false;
      return;
    }

    setFocusAlertSettings(readLocalFocusAlertSettings(user.uid));
    setFocusAlertSettingsLoading(true);
    const unsubscribe = subscribeToFocusAlertSettings(
      user.uid,
      (settings) => {
        setFocusAlertSettings(settings);
        saveLocalFocusAlertSettings(user.uid, {
          nudgeDelayMinutes: settings.nudgeDelayMinutes,
          soundEnabled: settings.soundEnabled,
          volume: settings.volume,
        });
        setFocusAlertSettingsLoading(false);
      },
      (error) => {
        const fallback = readLocalFocusAlertSettings(user.uid);
        if (isPermissionDenied(error)) {
          console.warn('Focus-alert settings are using local fallback storage because Firestore denied access.', error);
        } else {
          console.error('Error loading focus-alert settings:', error);
        }
        setFocusAlertSettings(fallback);
        setFocusAlertSettingsLoading(false);
        if (!shownLocalFallbackToastRef.current) {
          shownLocalFallbackToastRef.current = true;
          showToast(
            isPermissionDenied(error)
              ? 'Focus alerts are using this device only until Firestore rules are updated.'
              : 'Could not sync focus alerts. Using this device settings.',
            'info',
          );
        }
      },
    );

    return () => unsubscribe();
  }, [showToast, user]);

  const handleSaveFocusAlertSettings = async (
    nextSettings: Pick<FocusAlertSettings, 'nudgeDelayMinutes' | 'soundEnabled' | 'volume'>,
  ) => {
    if (!user) return;

    const previous = focusAlertSettings;
    const optimistic: FocusAlertSettings = {
      ...previous,
      ...nextSettings,
    };

    const localPersisted = saveLocalFocusAlertSettings(user.uid, nextSettings);
    setFocusAlertSettings({
      ...optimistic,
      ...localPersisted,
    });
    setFocusAlertSettingsSaving(true);
    try {
      await saveFocusAlertSettings(user.uid, nextSettings);
      showToast('Focus alerts updated.', 'success');
    } catch (error) {
      if (isPermissionDenied(error)) {
        console.warn('Saved focus-alert settings locally because Firestore denied access.', error);
        showToast('Focus alerts saved on this device only.', 'info');
        return;
      }
      console.error('Error saving focus-alert settings:', error);
      setFocusAlertSettings({
        ...previous,
        ...localPersisted,
      });
      showToast('Focus alerts saved on this device, but cloud sync failed.', 'warning');
    } finally {
      setFocusAlertSettingsSaving(false);
    }
  };

  return {
    focusAlertSettings,
    focusAlertSettingsLoading,
    focusAlertSettingsSaving,
    handleSaveFocusAlertSettings,
  };
}
