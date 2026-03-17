import { useState } from 'react';
import {
  ChevronUp,
  ChevronDown,
  LayoutDashboard,
  MessageSquareText,
  PanelRightOpen,
  Timer,
} from 'lucide-react';
import type { InferencePredictionPayload } from '../../services/inference-focus-tracker';

type MainView = 'chat' | 'dashboard';

type ChatMainHeaderProps = {
  mainView: MainView;
  currentChatName?: string;
  selectedModel: string;
  modelOptions: string[];
  focusBusy: boolean;
  isFocusActive: boolean;
  activeTrackerLabel: string | null;
  focusElapsedMs: number;
  lastPose: InferencePredictionPayload | null;
  isRightSidebarOpen: boolean;
  canToggleRightSidebar: boolean;
  onChangeModel: (value: string) => void;
  onChangeMainView: (nextView: MainView) => void;
  onToggleRightSidebar: () => void;
  formatDuration: (ms: number) => string;
};

export default function ChatMainHeader({
  mainView,
  currentChatName,
  selectedModel,
  modelOptions,
  focusBusy,
  isFocusActive,
  activeTrackerLabel,
  focusElapsedMs,
  lastPose,
  isRightSidebarOpen,
  canToggleRightSidebar,
  onChangeModel,
  onChangeMainView,
  onToggleRightSidebar,
  formatDuration,
}: ChatMainHeaderProps) {
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);
  const navBtnClass = (view: MainView) =>
    `chat-header-btn ${mainView === view ? 'chat-header-btn-primary' : ''}`;
  const subtitleText = mainView === 'dashboard'
    ? 'Visualize focus sessions captured from your webcam.'
    : (currentChatName ? '' : 'Choose an existing chat or create a new one to get started');

  return (
    <div className="chat-header">
      <div className="chat-header-inner">
        <div className="chat-header-left">
          <h2 className="chat-header-title">
            {mainView === 'dashboard' ? 'Focus dashboard' : (currentChatName || 'Select a Chat')}
          </h2>
          {subtitleText && <p className="chat-header-subtitle">{subtitleText}</p>}
          {mainView === 'chat' && (
            <div className="chat-header-model-row">
              <div className="chat-header-model-picker">
                <select
                  id="chat-header-model-select"
                  className="chat-header-model-select"
                  value={selectedModel}
                  onChange={(event) => {
                    onChangeModel(event.target.value);
                    setIsModelPickerOpen(false);
                  }}
                  onFocus={() => setIsModelPickerOpen(true)}
                  onBlur={() => setIsModelPickerOpen(false)}
                  onMouseDown={() => setIsModelPickerOpen(true)}
                  aria-label="Select AI model"
                >
                  {modelOptions.map((modelName) => (
                    <option key={modelName} value={modelName}>
                      {modelName}
                    </option>
                  ))}
                </select>
                {isModelPickerOpen ? (
                  <ChevronUp size={14} aria-hidden="true" className="chat-header-model-caret" />
                ) : (
                  <ChevronDown size={14} aria-hidden="true" className="chat-header-model-caret" />
                )}
              </div>
            </div>
          )}
        </div>

        <div className="chat-header-right">
          <div className="chat-header-controls">
            <div className="chat-header-row">
              <button
                onClick={() => onChangeMainView('chat')}
                className={navBtnClass('chat')}
                disabled={focusBusy}
                type="button"
              >
                <MessageSquareText size={16} aria-hidden="true" />
                <span>Chat</span>
              </button>
              <button
                onClick={() => onChangeMainView('dashboard')}
                className={navBtnClass('dashboard')}
                disabled={focusBusy}
                type="button"
              >
                <LayoutDashboard size={16} aria-hidden="true" />
                <span>Dashboard</span>
              </button>
              {!isRightSidebarOpen && (
                <button
                  onClick={onToggleRightSidebar}
                  className="chat-header-btn chat-header-btn-icon-only"
                  disabled={!canToggleRightSidebar}
                  type="button"
                  title="Show right panel"
                  aria-label="Show right panel"
                >
                  <PanelRightOpen size={16} aria-hidden="true" />
                </button>
              )}
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
