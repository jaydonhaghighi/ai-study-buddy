import type { InferencePredictionPayload } from '../../services/inference-focus-tracker';

type ChatMainHeaderProps = {
  mainView: 'chat' | 'dashboard';
  currentChatName?: string;
  focusBusy: boolean;
  isFocusActive: boolean;
  activeTrackerLabel: string | null;
  focusElapsedMs: number;
  lastPose: InferencePredictionPayload | null;
  settingsOpen: boolean;
  cameraPreviewEnabled: boolean;
  settingsRef: React.RefObject<HTMLDivElement>;
  settingsIconSrc: string;
  onToggleMainView: () => void;
  onStartFocus: () => void;
  onStopFocus: () => void;
  onToggleSettings: () => void;
  onToggleCameraPreview: () => void;
  onSignOut: () => void;
  formatDuration: (ms: number) => string;
};

export default function ChatMainHeader({
  mainView,
  currentChatName,
  focusBusy,
  isFocusActive,
  activeTrackerLabel,
  focusElapsedMs,
  lastPose,
  settingsOpen,
  cameraPreviewEnabled,
  settingsRef,
  settingsIconSrc,
  onToggleMainView,
  onStartFocus,
  onStopFocus,
  onToggleSettings,
  onToggleCameraPreview,
  onSignOut,
  formatDuration,
}: ChatMainHeaderProps) {
  return (
    <div className="chat-header">
      <div className="chat-header-inner">
        <div className="chat-header-left">
          <h2 className="chat-header-title">
            {mainView === 'dashboard' ? 'Focus dashboard' : (currentChatName || 'Select a Chat')}
          </h2>
          <p className="chat-header-subtitle">
            {mainView === 'dashboard'
              ? 'Visualize focus sessions captured from your webcam.'
              : (currentChatName ? 'AI Study Buddy' : 'Choose an existing chat or create a new one to get started')}
          </p>
        </div>

        <div className="chat-header-right">
          <div className="chat-header-controls">
            <div className="chat-header-row">
              <button
                onClick={onToggleMainView}
                className="chat-header-btn"
                disabled={focusBusy}
                type="button"
              >
                {mainView === 'dashboard' ? 'Back to chat' : 'Dashboard'}
              </button>

              {!isFocusActive ? (
                <button
                  onClick={onStartFocus}
                  className="chat-header-btn chat-header-btn-primary"
                  disabled={focusBusy}
                >
                  Start Focus
                </button>
              ) : (
                <button
                  onClick={onStopFocus}
                  className="chat-header-btn chat-header-btn-danger"
                  disabled={focusBusy}
                >
                  Stop Focus
                </button>
              )}

              <div className="chat-settings" ref={settingsRef}>
                <button
                  className="chat-header-btn chat-settings-btn"
                  type="button"
                  aria-label="Settings"
                  aria-haspopup="menu"
                  aria-expanded={settingsOpen}
                  onClick={onToggleSettings}
                  disabled={focusBusy}
                  title="Settings"
                >
                  <img src={settingsIconSrc} alt="" className="chat-settings-icon" />
                </button>

                {settingsOpen && (
                  <div className="chat-settings-menu" role="menu" aria-label="Settings menu">
                    <button
                      type="button"
                      className="chat-settings-item"
                      role="menuitem"
                      onClick={onToggleCameraPreview}
                      disabled={focusBusy}
                    >
                      Camera preview: {cameraPreviewEnabled ? 'On' : 'Off'}
                    </button>
                    <button
                      type="button"
                      className="chat-settings-item chat-settings-item-danger"
                      role="menuitem"
                      onClick={onSignOut}
                    >
                      Sign out
                    </button>
                  </div>
                )}
              </div>
            </div>

            {isFocusActive && (
              <div className="chat-header-meta">
                <span className="chat-header-pill">Focus active</span>
                <span>{activeTrackerLabel ?? 'Webcam tracking'}</span>
                <span>Duration: {formatDuration(focusElapsedMs)}</span>
                {lastPose && (
                  <span>
                    Pose: {lastPose.smoothed_label} ({Math.round(lastPose.smoothed_confidence * 100)}%)
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
