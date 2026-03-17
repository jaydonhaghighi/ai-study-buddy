import { useEffect, useMemo, useRef, useState } from 'react';
import { auth } from '../firebase-config';
import { User, signOut } from 'firebase/auth';
import FocusDashboard from './FocusDashboard';
import StudyMode from './StudyMode';
import ChatMainHeader from './chat/ChatMainHeader';
import ChatMessageList from './chat/ChatMessageList';
import ChatInput from './chat/ChatInput';
import ChatSidebar from './chat/ChatSidebar';
import ChatAuthView from './chat/ChatAuthView';
import ChatPreviewSidebar from './chat/ChatPreviewSidebar';
import ChatMaterialsPanel from './chat/ChatMaterialsPanel';
import ChatStudySetsPanel from './chat/ChatStudySetsPanel';
import ChatCalibrationModal from './chat/ChatCalibrationModal';
import { useChatAutoScroll } from './chat/useChatAutoScroll';
import { useFocusTracking } from './chat/useFocusTracking';
import { useChatCollections } from './chat/useChatCollections';
import { useChatMutations } from './chat/useChatMutations';
import { useChatAuth } from './chat/useChatAuth';
import { useChatCameraPreview } from './chat/useChatCameraPreview';
import { useChatUiState } from './chat/useChatUiState';
import { useChatMaterials } from './chat/useChatMaterials';
import { useStudySets } from './chat/useStudySets';
import logo from '../public/logo.png';
import './Chat.css';
import ExamContainer from './chat/ExamContainer';
import { ExamConfiguration } from '../types';
import ExamLauncher from './chat/ExamLauncher';

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
  const [mainView, setMainView] = useState<'chat' | 'dashboard' | 'exams'>('chat');
  
  const [expandedCourseId, setExpandedCourseId] = useState<string | null>(null);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [isChatSearchOpen, setIsChatSearchOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState('');
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);

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
    handleGoogleAuth,
  } = useChatAuth();

  // UI State
  const [isCreatingCourse, setIsCreatingCourse] = useState(false);
  const [newCourseName, setNewCourseName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editChatName, setEditChatName] = useState('');
  const audioCtxRef = useRef<AudioContext | null>(null);

  // Exam State
  const [activeExamConfig, setActiveExamConfig] = useState<ExamConfiguration | null>(null);
  const [examRunVersion, setExamRunVersion] = useState(0);

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
    currentFocusState,
    studyDistractions,
    firstDriftOffsetSec,
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
    previewPanelVisible: mainView === 'chat' && rightSidebarOpen,
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

  const currentChat = chats.find((chat) => chat.id === selectedChatId) ?? null;
  const {
    materials,
    materialsUploading,
    handleUploadMaterialFiles,
    handleDeleteMaterial,
  } = useChatMaterials({
    user,
    selectedChatId,
    currentChat,
    showToast,
  });

  const {
    studySets,
    activeStudySet,
    activeStudySetId,
    setActiveStudySetId,
    studySetGenerating,
    reviewBusyCardId,
    handleGenerateStudySet,
    handleReviewFlashcard,
  } = useStudySets({
    user,
    selectedChatId,
    currentChat,
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

  const visibleMessages = messages.filter((m) => {
    // Avoid rendering empty AI bubbles (e.g. streaming placeholder before first chunk).
    if (m.isAI && (!m.text || m.text.trim().length === 0)) return false;
    return true;
  });
  const hasStreamingAiText = messages.some(
    (m) => m.isAI && m.id.startsWith('temp-') && !!m.text && m.text.trim().length > 0
  );
  const showRightPanel = mainView === 'chat' && rightSidebarOpen;
  const normalizedChatQuery = chatSearchQuery.trim().toLowerCase();
  const filteredChatResults = useMemo(() => {
    if (!normalizedChatQuery) return chats;
    return chats.filter((chat) => chat.name.toLowerCase().includes(normalizedChatQuery));
  }, [chats, normalizedChatQuery]);

  const openChatSearchModal = () => {
    setChatSearchQuery('');
    setIsChatSearchOpen(true);
  };

  useEffect(() => {
    document.body.classList.toggle('left-sidebar-collapsed', !leftSidebarOpen);
    return () => {
      document.body.classList.remove('left-sidebar-collapsed');
    };
  }, [leftSidebarOpen]);

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
        onGoogleSignIn={handleGoogleAuth}
      />
    );
  }

  return (
    <div className={`chat-container ${showRightPanel ? 'preview-sidebar-open' : ''} ${leftSidebarOpen ? '' : 'left-sidebar-closed'}`}>
      {leftSidebarOpen && (
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
          onCloseSidebar={() => setLeftSidebarOpen(false)}
        />
      )}
      {!leftSidebarOpen && (
        <aside className="chat-sidebar-rail" aria-label="Collapsed navigation">
          <button
            type="button"
            className="chat-sidebar-rail-logo"
            onClick={() => setLeftSidebarOpen(true)}
            title="Expand courses panel"
            aria-label="Expand courses panel"
          >
            <img src={logo} alt="Echelon logo" />
          </button>
          <div className="chat-sidebar-rail-actions">
            <button
              type="button"
              className="chat-sidebar-rail-btn"
              onClick={() => {
                setLeftSidebarOpen(true);
                setIsCreatingCourse(true);
              }}
              title="Add course"
              aria-label="Add course"
            >
              +
            </button>
            <button
              type="button"
              className="chat-sidebar-rail-btn"
              onClick={openChatSearchModal}
              title="Search chats"
              aria-label="Search chats"
            >
              ?
            </button>
            <button
              type="button"
              className="chat-sidebar-rail-btn"
              onClick={() => setLeftSidebarOpen(true)}
              title="Expand panel"
              aria-label="Expand panel"
            >
              &gt;&gt;
            </button>
          </div>
        </aside>
      )}

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
          isLeftSidebarOpen={leftSidebarOpen}
          isRightSidebarOpen={rightSidebarOpen}
          canToggleRightSidebar={mainView === 'chat'}
          settingsRef={settingsRef}
          onToggleLeftSidebar={() => setLeftSidebarOpen((v) => !v)}
          onToggleRightSidebar={() => setRightSidebarOpen((v) => !v)}
          onChangeMainView={setMainView}
          onStartFocus={handleStartFocus}
          onStartExams={() => setMainView('exams')}
          onStopFocus={handleStopFocus}
          onToggleSettings={() => setSettingsOpen((v) => !v)}
          onToggleCameraPreview={() => setCameraPreviewEnabled((v) => !v)}
          onSignOut={handleSignOut}
          formatDuration={formatDuration}
        />

        {activeExamConfig ? (
          <ExamContainer
            key={`${activeExamConfig.id}-${examRunVersion}`}
            examConfig={activeExamConfig}
            userId={user.uid}
            onComplete={() => {}}
            onExit={() => {
              setActiveExamConfig(null);
              setMainView('exams');
            }}
            onRestart={() => setExamRunVersion((value) => value + 1)}
          />
        ) : mainView === 'exams' ? (
          <ExamLauncher
            userId={user.uid}
            selectedChatId={selectedChatId}
            onSelectChat={setSelectedChatId}
            chats={chats}
            activeStudySet={activeStudySet}
            studySets={studySets}
            onSelectExam={(exam) => {
              setExamRunVersion(0);
              setActiveExamConfig(exam);
            }}
          />
        ) : mainView === 'dashboard' ? (
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
        show={showRightPanel}
        studyContent={(
          <StudyMode
            variant="sidebar"
            userId={user.uid}
            activeFocusSession={activeFocusSession}
            focusBusy={focusBusy}
            currentFocusState={currentFocusState}
            studyDistractions={studyDistractions}
            firstDriftOffsetSec={firstDriftOffsetSec}
            showCalibrationModal={showCalibrationModal}
            onStartFocus={handleStartFocus}
            onStopFocus={handleStopFocus}
            showToast={showToast}
          />
        )}
        materialsContent={(
          <ChatMaterialsPanel
            selectedChatId={selectedChatId}
            materials={materials}
            materialsUploading={materialsUploading}
            onUploadFiles={handleUploadMaterialFiles}
            onDeleteMaterial={handleDeleteMaterial}
          />
        )}
        activeRecallContent={(
          <ChatStudySetsPanel
            selectedChatId={selectedChatId}
            studySets={studySets}
            activeStudySet={activeStudySet}
            activeStudySetId={activeStudySetId}
            studySetGenerating={studySetGenerating}
            reviewBusyCardId={reviewBusyCardId}
            onSelectStudySet={setActiveStudySetId}
            onGenerateStudySet={handleGenerateStudySet}
            onReviewFlashcard={handleReviewFlashcard}
          />
        )}
        isLocalTrackerRunning={isLocalTrackerRunning}
        cameraPreviewEnabled={cameraPreviewEnabled}
        cameraPreviewAfterCalibration={cameraPreviewAfterCalibration}
        previewError={previewError}
        previewVideoRef={previewVideoRef}
        onClose={() => setRightSidebarOpen(false)}
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

      {isChatSearchOpen && (
        <div
          className="chat-search-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Search chats"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setIsChatSearchOpen(false);
            }
          }}
        >
          <div className="chat-search-modal">
            <div className="chat-search-modal-header">
              <h3>Search Chats</h3>
              <button
                type="button"
                className="chat-search-modal-close"
                onClick={() => setIsChatSearchOpen(false)}
                aria-label="Close chat search"
              >
                x
              </button>
            </div>

            <div className="chat-search-input-wrap">
              <span aria-hidden="true">Search</span>
              <input
                type="text"
                className="chat-search-input"
                placeholder="Search by chat name..."
                value={chatSearchQuery}
                onChange={(event) => setChatSearchQuery(event.target.value)}
                autoFocus
              />
            </div>

            <div className="chat-search-results">
              {filteredChatResults.length === 0 ? (
                <p className="chat-search-empty">
                  {chats.length === 0
                    ? 'No chats loaded yet. Expand a course and session first.'
                    : 'No chats match your search.'}
                </p>
              ) : (
                filteredChatResults.map((chat) => (
                  <button
                    key={chat.id}
                    type="button"
                    className={`chat-search-result-item ${selectedChatId === chat.id ? 'active' : ''}`}
                    onClick={() => {
                      setSelectedChatId(chat.id);
                      setIsChatSearchOpen(false);
                    }}
                  >
                    {chat.name}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
