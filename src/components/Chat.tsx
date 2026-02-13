import { useState, useRef } from 'react';
import { auth } from '../firebase-config';
import { User, signOut } from 'firebase/auth';
import FocusDashboard from './FocusDashboard';
import ChatMainHeader from './chat/ChatMainHeader';
import ChatMessageList from './chat/ChatMessageList';
import ChatInput from './chat/ChatInput';
import ChatSidebar from './chat/ChatSidebar';
import ChatAuthView from './chat/ChatAuthView';
import ChatPreviewSidebar from './chat/ChatPreviewSidebar';
import ChatCalibrationModal from './chat/ChatCalibrationModal';
import { useChatAutoScroll } from './chat/useChatAutoScroll';
import { useFocusTracking } from './chat/useFocusTracking';
import { useChatCollections } from './chat/useChatCollections';
import { useChatMutations } from './chat/useChatMutations';
import { useChatAuth } from './chat/useChatAuth';
import { useChatCameraPreview } from './chat/useChatCameraPreview';
import { useChatUiState } from './chat/useChatUiState';
import settingsIcon from '../public/settings.svg';
import './Chat.css';

interface ChatProps {
  user: User | null;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

export default function Chat({ user }: ChatProps) {
  // Navigation State
  const [mainView, setMainView] = useState<'chat' | 'dashboard'>('chat');
  
  const [expandedCourseId, setExpandedCourseId] = useState<string | null>(null);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);

  const {
    courses,
    sessions,
    chats,
    messages,
    setMessages,
    clearAllData,
  } = useChatCollections({
    user,
    expandedCourseId,
    expandedSessionId,
    selectedChatId,
    mainView,
  });

  // Chat State
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const {
    showAuth,
    setShowAuth,
    authEmail,
    setAuthEmail,
    authPassword,
    setAuthPassword,
    isSignUp,
    setIsSignUp,
    authError,
    authLoading,
    handleAuth,
  } = useChatAuth();

  // UI State
  const [isCreatingCourse, setIsCreatingCourse] = useState(false);
  const [newCourseName, setNewCourseName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editChatName, setEditChatName] = useState('');
  const audioCtxRef = useRef<AudioContext | null>(null);
  const {
    toastMessage,
    toastVariant,
    showToast,
    settingsOpen,
    setSettingsOpen,
    settingsRef,
  } = useChatUiState();

  const playBeep = (frequencyHz: number, durationMs: number, gain: number = 0.06) => {
    try {
      const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextCtor) return;
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContextCtor();
      }
      const ctx = audioCtxRef.current;
      // best-effort resume; may still be blocked until user gesture (Start Focus counts)
      if (ctx.state === 'suspended') {
        void ctx.resume().catch(() => {});
      }
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = frequencyHz;
      g.gain.value = gain;
      osc.connect(g);
      g.connect(ctx.destination);
      const now = ctx.currentTime;
      osc.start(now);
      osc.stop(now + durationMs / 1000);
    } catch {
      // ignore audio failures (permissions/autoplay policy)
    }
  };

  const playFocusTransitionSound = (nextState: 'focused' | 'distracted') => {
    if (nextState === 'distracted') {
      playBeep(220, 140);
      window.setTimeout(() => playBeep(220, 140), 170);
    } else {
      playBeep(660, 180);
    }
  };

  const {
    activeFocusSession,
    focusBusy,
    isLocalTrackerRunning,
    activeTrackerLabel,
    lastPose,
    focusElapsedMs,
    showCalibrationModal,
    handleStartFocus,
    startFocusAfterCalibration,
    handleStopFocus,
    hideCalibrationModal,
    cancelCalibration,
    cleanupOnSignOut,
  } = useFocusTracking({
    user,
    expandedCourseId,
    expandedSessionId,
    showToast,
    playFocusTransitionSound,
  });

  const {
    cameraPreviewEnabled,
    setCameraPreviewEnabled,
    cameraPreviewAfterCalibration,
    setCameraPreviewAfterCalibration,
    previewVideoRef,
    previewError,
  } = useChatCameraPreview({
    mainView,
    showCalibrationModal,
  });

  const {
    messagesEndRef,
    messagesContainerRef,
    handleMessagesScroll,
    enableAutoScroll,
  } = useChatAutoScroll({
    selectedChatId,
    mainView,
    messages,
    loading,
  });

  const {
    handleCreateCourse,
    handleCreateSession,
    handleCreateChat,
    handleSend,
    handleUpdateChatName,
    handleDeleteChat,
  } = useChatMutations({
    user,
    expandedCourseId,
    expandedSessionId,
    selectedChatId,
    input,
    newCourseName,
    newSessionName,
    setInput,
    setLoading,
    setMessages,
    setNewCourseName,
    setIsCreatingCourse,
    setNewSessionName,
    setIsCreatingSession,
    setExpandedSessionId,
    setSelectedChatId,
    setEditingChatId,
    enableAutoScroll,
    showToast,
  });

  const handleSignOut = async () => {
    try {
      setSettingsOpen(false);
      await cleanupOnSignOut();
      await signOut(auth);
      clearAllData();
      setSelectedChatId(null);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  if (!user) {
    return (
      <ChatAuthView
        showAuth={showAuth}
        authEmail={authEmail}
        authPassword={authPassword}
        isSignUp={isSignUp}
        authError={authError}
        authLoading={authLoading}
        onShowAuth={setShowAuth}
        onEmailChange={setAuthEmail}
        onPasswordChange={setAuthPassword}
        onToggleSignUp={() => setIsSignUp(!isSignUp)}
        onSubmit={handleAuth}
      />
    );
  }

  const currentChat = chats.find(c => c.id === selectedChatId);
  const visibleMessages = messages.filter((m) => {
    // Avoid rendering empty AI bubbles (e.g. streaming placeholder before first chunk).
    if (m.isAI && (!m.text || m.text.trim().length === 0)) return false;
    return true;
  });
  const hasStreamingAiText = messages.some(
    (m) => m.isAI && m.id.startsWith('temp-') && !!m.text && m.text.trim().length > 0
  );

  return (
    <div className={`chat-container ${cameraPreviewEnabled && mainView === 'chat' ? 'preview-sidebar-open' : ''}`}>
      <ChatSidebar
        courses={courses}
        sessions={sessions}
        chats={chats}
        expandedCourseId={expandedCourseId}
        expandedSessionId={expandedSessionId}
        selectedChatId={selectedChatId}
        isCreatingCourse={isCreatingCourse}
        newCourseName={newCourseName}
        isCreatingSession={isCreatingSession}
        newSessionName={newSessionName}
        editingChatId={editingChatId}
        editChatName={editChatName}
        onSetCreatingCourse={setIsCreatingCourse}
        onSetNewCourseName={setNewCourseName}
        onCreateCourse={handleCreateCourse}
        onToggleCourse={(courseId) => setExpandedCourseId(expandedCourseId === courseId ? null : courseId)}
        onSetCreatingSession={setIsCreatingSession}
        onSetNewSessionName={setNewSessionName}
        onCreateSession={handleCreateSession}
        onToggleSession={(sessionId) => setExpandedSessionId(expandedSessionId === sessionId ? null : sessionId)}
        onCreateChat={handleCreateChat}
        onSelectChat={setSelectedChatId}
        onSetEditingChatId={setEditingChatId}
        onSetEditChatName={setEditChatName}
        onUpdateChatName={handleUpdateChatName}
        onDeleteChat={handleDeleteChat}
      />

      {/* Main Area */}
      <div className="chat-main">
        <ChatMainHeader
          mainView={mainView}
          currentChatName={currentChat?.name}
          focusBusy={focusBusy}
          isFocusActive={!!activeFocusSession}
          activeTrackerLabel={activeTrackerLabel}
          focusElapsedMs={focusElapsedMs}
          lastPose={lastPose}
          settingsOpen={settingsOpen}
          cameraPreviewEnabled={cameraPreviewEnabled}
          settingsRef={settingsRef}
          settingsIconSrc={settingsIcon}
          onToggleMainView={() => setMainView(mainView === 'dashboard' ? 'chat' : 'dashboard')}
          onStartFocus={handleStartFocus}
          onStopFocus={handleStopFocus}
          onToggleSettings={() => setSettingsOpen((v) => !v)}
          onToggleCameraPreview={() => setCameraPreviewEnabled((v) => !v)}
          onSignOut={handleSignOut}
          formatDuration={formatDuration}
        />

        {mainView === 'dashboard' ? (
          <FocusDashboard userId={user.uid} />
        ) : (
          <>
            <ChatMessageList
              selectedChatId={selectedChatId}
              visibleMessages={visibleMessages}
              loading={loading}
              hasStreamingAiText={hasStreamingAiText}
              messagesContainerRef={messagesContainerRef}
              messagesEndRef={messagesEndRef}
              onMessagesScroll={handleMessagesScroll}
            />

            <ChatInput
              selectedChatId={selectedChatId}
              input={input}
              loading={loading}
              onInputChange={setInput}
              onSubmit={handleSend}
            />
          </>
        )}
      </div>

      <ChatPreviewSidebar
        show={cameraPreviewEnabled && mainView === 'chat'}
        isLocalTrackerRunning={isLocalTrackerRunning}
        cameraPreviewAfterCalibration={cameraPreviewAfterCalibration}
        previewError={previewError}
        previewVideoRef={previewVideoRef}
      />
      
      {toastMessage && (
        <div className={`toast-notification ${toastVariant === 'warning' ? 'toast-warning' : toastVariant === 'info' ? 'toast-info' : ''}`}>
          {toastMessage}
        </div>
      )}

      <ChatCalibrationModal
        show={showCalibrationModal}
        onCancel={cancelCalibration}
        onAlignedStable={() => {
          hideCalibrationModal();
          setCameraPreviewAfterCalibration(true);
          startFocusAfterCalibration();
        }}
      />
    </div>
  );
}
