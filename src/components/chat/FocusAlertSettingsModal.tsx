import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import type { FocusAlertSettings } from '../../types';
import {
  MAX_FOCUS_ALERT_DELAY_MINUTES,
  MIN_FOCUS_ALERT_DELAY_MINUTES,
} from '../../services/focus-alert-settings-service';

type FocusAlertSettingsDraft = Pick<FocusAlertSettings, 'nudgeDelayMinutes' | 'soundEnabled' | 'volume'>;

type FocusAlertSettingsModalProps = {
  show: boolean;
  settings: FocusAlertSettings;
  loading: boolean;
  saving: boolean;
  onClose: () => void;
  onPreview: (volume: number) => void;
  onSave: (settings: FocusAlertSettingsDraft) => Promise<void>;
};

function formatMinutesLabel(minutes: number): string {
  return `${minutes} minute${minutes === 1 ? '' : 's'}`;
}

export default function FocusAlertSettingsModal({
  show,
  settings,
  loading,
  saving,
  onClose,
  onPreview,
  onSave,
}: FocusAlertSettingsModalProps) {
  const [draft, setDraft] = useState<FocusAlertSettingsDraft>({
    nudgeDelayMinutes: settings.nudgeDelayMinutes,
    soundEnabled: settings.soundEnabled,
    volume: settings.volume,
  });

  useEffect(() => {
    if (!show) return;
    setDraft({
      nudgeDelayMinutes: settings.nudgeDelayMinutes,
      soundEnabled: settings.soundEnabled,
      volume: settings.volume,
    });
  }, [settings, show]);

  if (!show) return null;

  const volumePercent = Math.round(draft.volume * 100);
  const disabled = loading || saving;

  return (
    <div
      className="modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Focus alert settings"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !saving) {
          onClose();
        }
      }}
    >
      <div className="modal focus-alert-modal">
        <div className="modal-header">
          <div>
            <h3 className="modal-title">Focus alerts</h3>
            <p className="modal-subtitle">
              Choose when a sound reminder should play after you drift away from the screen.
            </p>
          </div>
          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
            type="button"
            disabled={saving}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <form
          className="modal-body focus-alert-form"
          onSubmit={(event) => {
            event.preventDefault();
            void (async () => {
              try {
                await onSave(draft);
                onClose();
              } catch {
                // The save handler already shows a toast; keep the modal open.
              }
            })();
          }}
        >
          <div className="focus-alert-fieldset">
            <div className="focus-alert-field-header">
              <label htmlFor="focus-alert-delay" className="focus-alert-label">
                Nudge me after {formatMinutesLabel(draft.nudgeDelayMinutes)} unfocused
              </label>
              <span className="focus-alert-value">{draft.nudgeDelayMinutes} min</span>
            </div>
            <input
              id="focus-alert-delay"
              className="focus-alert-range"
              type="range"
              min={MIN_FOCUS_ALERT_DELAY_MINUTES}
              max={MAX_FOCUS_ALERT_DELAY_MINUTES}
              step={1}
              value={draft.nudgeDelayMinutes}
              onChange={(event) => {
                setDraft((prev) => ({
                  ...prev,
                  nudgeDelayMinutes: Number(event.target.value),
                }));
              }}
              disabled={disabled}
            />
            <div className="focus-alert-scale" aria-hidden="true">
              <span>{MIN_FOCUS_ALERT_DELAY_MINUTES} min</span>
              <span>{MAX_FOCUS_ALERT_DELAY_MINUTES} min</span>
            </div>
          </div>

          <label className={`focus-alert-toggle ${draft.soundEnabled ? '' : 'focus-alert-toggle--muted'}`}>
            <input
              type="checkbox"
              checked={draft.soundEnabled}
              onChange={(event) => {
                setDraft((prev) => ({
                  ...prev,
                  soundEnabled: event.target.checked,
                }));
              }}
              disabled={disabled}
            />
            <div>
              <span className="focus-alert-label">Sound alerts</span>
              <p className="focus-alert-copy">
                Plays one reminder per distraction streak. The timer resets as soon as you refocus.
              </p>
            </div>
          </label>

          <div className={`focus-alert-fieldset ${draft.soundEnabled ? '' : 'focus-alert-fieldset--disabled'}`}>
            <div className="focus-alert-field-header">
              <label htmlFor="focus-alert-volume" className="focus-alert-label">
                Nudge volume
              </label>
              <span className="focus-alert-value">{volumePercent}%</span>
            </div>
            <input
              id="focus-alert-volume"
              className="focus-alert-range"
              type="range"
              min={0}
              max={100}
              step={5}
              value={volumePercent}
              onChange={(event) => {
                setDraft((prev) => ({
                  ...prev,
                  volume: Number(event.target.value) / 100,
                }));
              }}
              disabled={disabled || !draft.soundEnabled}
            />
            <div className="focus-alert-scale" aria-hidden="true">
              <span>Quiet</span>
              <span>Loud</span>
            </div>
          </div>

          <div className="modal-hint">
            Focus tracking still shows immediate state toasts. This setting controls only the delayed sound nudge.
          </div>

          <div className="focus-alert-actions">
            <button
              className="modal-btn"
              type="button"
              onClick={() => onPreview(draft.volume)}
              disabled={disabled || !draft.soundEnabled}
            >
              Preview sound
            </button>
            <div className="focus-alert-actions-spacer" />
            <button className="modal-btn" type="button" onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button className="modal-btn modal-btn-primary" type="submit" disabled={disabled}>
              {saving ? 'Saving...' : 'Save settings'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
