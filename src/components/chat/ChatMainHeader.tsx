import {
  Camera,
  CameraOff,
  LayoutDashboard,
  LogOut,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Settings,
  Timer,
} from 'lucide-react';
import type { InferencePredictionPayload } from '../../services/inference-focus-tracker';

type MainView = 'chat' | 'dashboard';

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
  isLeftSidebarOpen: boolean;
  isRightSidebarOpen: boolean;
  canToggleRightSidebar: boolean;
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
  isLeftSidebarOpen,
  isRightSidebarOpen,
  canToggleRightSidebar,
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
  const navBtnClass = (view: MainView) =>
    `chat-header-btn ${mainView === view ? 'chat-header-btn-primary' : ''}`;

  return (
    <div className="chat-header">
      <div className="chat-header-inner">
        <div className="chat-header-left">
          <h2 className="chat-header-title">
            {mainView === 'exams'
              ? 'Exam Simulations'
              : mainView === 'dashboard'
              ? 'Focus Dashboard'
              : currentChatName || 'Select a Chat'}
          </h2>
          <p className="chat-header-subtitle">
            {mainView === 'exams'
              ? 'Practice with timed mock exams to prepare for assessments.'
              : mainView === 'dashboard'
              ? 'Visualize focus sessions captured from your webcam.'
              : (currentChatName ? 'AI Study Buddy' : 'Choose an existing chat or create a new one to get started')}
          </p>
        </div>

        <div className="chat-header-right">
          <div className="chat-header-controls">
            <div className="chat-header-row">
              <button
                onClick={onToggleLeftSidebar}
                className="chat-header-btn"
                type="button"
                title={isLeftSidebarOpen ? 'Hide left panel' : 'Show left panel'}
              >
                {isLeftSidebarOpen ? <PanelLeftClose size={16} aria-hidden="true" /> : <PanelLeftOpen size={16} aria-hidden="true" />}
                <span>{isLeftSidebarOpen ? 'Hide Left' : 'Show Left'}</span>
              </button>
              <button
                onClick={onToggleRightSidebar}
                className="chat-header-btn"
                disabled={!canToggleRightSidebar}
                type="button"
                title={isRightSidebarOpen ? 'Hide right panel' : 'Show right panel'}
              >
                {isRightSidebarOpen ? <PanelRightClose size={16} aria-hidden="true" /> : <PanelRightOpen size={16} aria-hidden="true" />}
                <span>{isRightSidebarOpen ? 'Hide Right' : 'Show Right'}</span>
              </button>
              <button
                onClick={() => onChangeMainView('chat')}
                className={navBtnClass('chat')}
                disabled={focusBusy}
                type="button"
              >
                <MessageSquareText size={16} aria-hidden="true" />
                <span>Chat</span>
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
                  <Settings size={18} aria-hidden="true" />
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
                      {cameraPreviewEnabled ? <Camera size={16} aria-hidden="true" /> : <CameraOff size={16} aria-hidden="true" />}
                      <span>Camera preview: {cameraPreviewEnabled ? 'On' : 'Off'}</span>
                    </button>
                    <button
                      type="button"
                      className="chat-settings-item chat-settings-item-danger"
                      role="menuitem"
                      onClick={onSignOut}
                    >
                      <LogOut size={16} aria-hidden="true" />
                      <span>Sign out</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            {isFocusActive && (
              <div className="chat-header-meta">
                <span className="chat-header-pill">Focus active</span>
                <span>{activeTrackerLabel ?? 'Webcam tracking'}</span>
                <span className="chat-header-meta-inline">
                  <Timer size={14} aria-hidden="true" />
                  <span>Duration: {formatDuration(focusElapsedMs)}</span>
                </span>
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
