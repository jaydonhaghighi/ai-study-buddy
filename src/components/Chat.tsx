import { useEffect, useMemo, useState } from 'react';
import { auth } from '../firebase-config';
import { User, signOut } from 'firebase/auth';
import {
  Bell,
  Camera,
  FolderPlus,
  LogOut,
  PanelLeftOpen,
  Search,
  Settings,
  X,
} from 'lucide-react';
import FocusDashboard from './FocusDashboard';
import ExamSimulationWorkspace from './ExamSimulationWorkspace';
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
import FocusAlertSettingsModal from './chat/FocusAlertSettingsModal';
import { useChatAutoScroll } from './chat/useChatAutoScroll';
import { useFocusTracking } from './chat/useFocusTracking';
import { useChatCollections } from './chat/useChatCollections';
import { useChatMutations } from './chat/useChatMutations';
import { useChatAuth } from './chat/useChatAuth';
import { useChatCameraPreview } from './chat/useChatCameraPreview';
import { useFocusAlertAudio } from './chat/useFocusAlertAudio';
import { useFocusAlertSettings } from './chat/useFocusAlertSettings';
import { useChatUiState } from './chat/useChatUiState';
import { useChatMaterials } from './chat/useChatMaterials';
import { useStudySets } from './chat/useStudySets';
import { useExamSimulation } from './chat/useExamSimulation';
import logo from '../public/logo.png';
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

const DEFAULT_OPENAI_CHAT_MODEL = 'gpt-4o-mini';
const FALLBACK_OPENAI_CHAT_MODELS = [
  'gpt-4o-mini',
  'gpt-4o',
  'gpt-4.1-nano',
  'gpt-4.1-mini',
  'gpt-4.1',
  'gpt-5-nano',
  'gpt-5-mini',
  'gpt-5',
  'gpt-5.1',
  'gpt-5.2',
  'gpt-5.4',
];
const CHAT_MODEL_STORAGE_KEY = 'studybuddy.chatModel';

function parseModelList(raw: string | undefined): string[] {
  const source = (raw ?? '').trim();
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of source.split(',')) {
    const modelName = token.trim();
    if (!modelName || seen.has(modelName)) continue;
    seen.add(modelName);
    out.push(modelName);
  }
  return out;
}

const configuredModelOptions = parseModelList(import.meta.env.VITE_OPENAI_CHAT_MODELS);
const OPENAI_CHAT_MODEL_OPTIONS = configuredModelOptions.length > 0
  ? configuredModelOptions
  : FALLBACK_OPENAI_CHAT_MODELS;

export default function Chat({ user }: ChatProps) {
  // Navigation State
  const [mainView, setMainView] = useState<'chat' | 'exam' | 'dashboard'>('chat');
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);
  
  const [expandedCourseId, setExpandedCourseId] = useState<string | null>(null);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [isChatSearchOpen, setIsChatSearchOpen] = useState(false);
  const [isFocusAlertSettingsOpen, setIsFocusAlertSettingsOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState('');

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
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    const fallbackModel = OPENAI_CHAT_MODEL_OPTIONS[0] ?? DEFAULT_OPENAI_CHAT_MODEL;
    if (typeof window === 'undefined') return fallbackModel;
    const storedModel = window.localStorage.getItem(CHAT_MODEL_STORAGE_KEY);
    if (storedModel && OPENAI_CHAT_MODEL_OPTIONS.includes(storedModel)) {
      return storedModel;
    }
    return fallbackModel;
  });

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

  useEffect(() => {
    if (OPENAI_CHAT_MODEL_OPTIONS.includes(selectedModel)) return;
    setSelectedModel(OPENAI_CHAT_MODEL_OPTIONS[0] ?? DEFAULT_OPENAI_CHAT_MODEL);
  }, [selectedModel]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(CHAT_MODEL_STORAGE_KEY, selectedModel);
  }, [selectedModel]);

  // UI State
  const [isCreatingCourse, setIsCreatingCourse] = useState(false);
  const [newCourseName, setNewCourseName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editChatName, setEditChatName] = useState('');
  const {
    toastMessage,
    toastVariant,
    showToast,
    settingsOpen,
    setSettingsOpen,
    settingsRef,
  } = useChatUiState();

  const {
    focusAlertSettings,
    focusAlertSettingsLoading,
    focusAlertSettingsSaving,
    handleSaveFocusAlertSettings,
  } = useFocusAlertSettings({
    user,
    showToast,
  });

  const {
    playFocusRecoverySound,
    playDistractedNudgeSound,
    playFocusAlertPreviewSound,
  } = useFocusAlertAudio({
    soundEnabled: focusAlertSettings.soundEnabled,
    volume: focusAlertSettings.volume,
  });

  const focusAlertStatusLabel = focusAlertSettings.soundEnabled
    ? `${focusAlertSettings.nudgeDelayMinutes} min`
    : 'Off';

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
    focusAlertSettings,
    showToast,
    playFocusRecoverySound,
    playDistractedNudgeSound,
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
    previewPanelVisible: mainView === 'chat',
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
    selectedModel,
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
    selectedModel,
    showToast,
  });

  const {
    examSimulations,
    activeExam,
    activeExamId,
    setActiveExamId,
    examGenerating,
    examActionBusy,
    handleCreateExam,
    handleStartExam,
    handleSubmitAnswer,
    handleFinishExam,
  } = useExamSimulation({
    user,
    selectedChatId,
    currentChat,
    selectedModel,
    showToast,
  });

  const handleSignOut = async () => {
    try {
      setSettingsOpen(false);
      setIsFocusAlertSettingsOpen(false);
      await cleanupOnSignOut();
      await signOut(auth);
      clearAllData();
      setSelectedChatId(null);
    } catch (error) {
      console.error("Sign out error:", error);
    }
  };

  const normalizedChatQuery = chatSearchQuery.trim().toLowerCase();
  const filteredChatResults = useMemo(() => {
    if (!normalizedChatQuery) return chats;
    return chats.filter((chat) => chat.name.toLowerCase().includes(normalizedChatQuery));
  }, [chats, normalizedChatQuery]);

  const openChatSearchModal = () => {
    setChatSearchQuery('');
    setIsChatSearchOpen(true);
  };

  const openFocusAlertSettings = () => {
    setSettingsOpen(false);
    setIsFocusAlertSettingsOpen(true);
  };

  useEffect(() => {
    if (!user) {
      document.body.classList.remove('left-sidebar-collapsed');
      return;
    }
    document.body.classList.toggle('left-sidebar-collapsed', !leftSidebarOpen);
    return () => {
      document.body.classList.remove('left-sidebar-collapsed');
    };
  }, [leftSidebarOpen, user]);

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

  const visibleMessages = messages.filter((m) => {
    // Avoid rendering empty AI bubbles (e.g. streaming placeholder before first chunk).
    if (m.isAI && (!m.text || m.text.trim().length === 0)) return false;
    return true;
  });
  const hasStreamingAiText = messages.some(
    (m) => m.isAI && m.id.startsWith('temp-') && !!m.text && m.text.trim().length > 0
  );
  const examIsLocked = mainView === 'exam' && activeExam?.status === 'in_progress';
  const showRightPanel = mainView === 'chat' && rightSidebarOpen && !examIsLocked;

  return (
    <div className={`chat-container ${showRightPanel ? 'preview-sidebar-open' : ''} ${leftSidebarOpen ? '' : 'left-sidebar-closed'}`}>
      {!examIsLocked && leftSidebarOpen && (
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
          settingsOpen={settingsOpen}
          settingsRef={settingsRef}
          focusAlertStatusLabel={focusAlertStatusLabel}
          cameraPreviewEnabled={cameraPreviewEnabled}
          focusBusy={focusBusy}
          onToggleSettings={() => setSettingsOpen((value) => !value)}
          onOpenFocusAlerts={openFocusAlertSettings}
          onToggleCameraPreview={() => setCameraPreviewEnabled((value) => !value)}
          onSignOut={handleSignOut}
        />
      )}
      {!examIsLocked && !leftSidebarOpen && (
        <aside className="chat-sidebar-rail" aria-label="Collapsed navigation">
          <button
            type="button"
            className="chat-sidebar-rail-logo"
            onClick={() => setLeftSidebarOpen(true)}
            title="Expand courses panel"
            aria-label="Expand courses panel"
          >
            <img src={logo} alt="Echelon logo" />
            <span className="chat-sidebar-rail-logo-overlay" aria-hidden="true">
              <PanelLeftOpen size={18} />
            </span>
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
              <FolderPlus size={18} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="chat-sidebar-rail-btn"
              onClick={openChatSearchModal}
              title="Search chats"
              aria-label="Search chats"
            >
              <Search size={18} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="chat-sidebar-rail-btn"
              onClick={() => setLeftSidebarOpen(true)}
              title="Expand panel"
              aria-label="Expand panel"
            >
              <PanelLeftOpen size={18} aria-hidden="true" />
            </button>
          </div>
          <div className="chat-sidebar-rail-footer">
            <div className="chat-settings" ref={settingsRef}>
              <button
                className="chat-sidebar-rail-btn"
                type="button"
                aria-label="Settings"
                aria-haspopup="menu"
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((value) => !value)}
                disabled={focusBusy}
                title="Settings"
              >
                <Settings size={18} aria-hidden="true" />
              </button>

              {settingsOpen && (
                <div className="chat-settings-menu chat-settings-menu-rail" role="menu" aria-label="Settings menu">
                  <button
                    type="button"
                    className="chat-settings-item"
                    role="menuitem"
                    onClick={openFocusAlertSettings}
                  >
                    <Bell size={16} aria-hidden="true" />
                    <span>Focus alerts: {focusAlertStatusLabel}</span>
                  </button>
                  <button
                    type="button"
                    className="chat-settings-item"
                    role="menuitem"
                    onClick={() => setCameraPreviewEnabled((value) => !value)}
                    disabled={focusBusy}
                  >
                    <Camera size={16} aria-hidden="true" />
                    <span>Camera preview: {cameraPreviewEnabled ? 'On' : 'Off'}</span>
                  </button>
                  <button
                    type="button"
                    className="chat-settings-item chat-settings-item-danger"
                    role="menuitem"
                    onClick={handleSignOut}
                  >
                    <LogOut size={16} aria-hidden="true" />
                    <span>Sign out</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </aside>
      )}

      {/* Main Area */}
      <div className="chat-main">
        <ChatMainHeader
          mainView={mainView}
          currentChatName={currentChat?.name}
          selectedModel={selectedModel}
          modelOptions={OPENAI_CHAT_MODEL_OPTIONS}
          focusBusy={focusBusy}
          isFocusActive={!!activeFocusSession}
          activeTrackerLabel={activeTrackerLabel}
          focusElapsedMs={focusElapsedMs}
          lastPose={lastPose}
          onChangeModel={setSelectedModel}
          onChangeMainView={setMainView}
          isRightSidebarOpen={rightSidebarOpen}
          canToggleRightSidebar={mainView === 'chat' && !examIsLocked}
          isExamLocked={examIsLocked}
          onToggleRightSidebar={() => setRightSidebarOpen((value) => !value)}
          formatDuration={formatDuration}
        />

        {mainView === 'dashboard' ? (
          <FocusDashboard userId={user.uid} />
        ) : mainView === 'exam' ? (
          <ExamSimulationWorkspace
            selectedChatId={selectedChatId}
            currentChat={currentChat}
            indexedMaterialsCount={materials.filter((material) => material.status === 'indexed').length}
            selectedModel={selectedModel}
            examSimulations={examSimulations}
            activeExam={activeExam}
            activeExamId={activeExamId}
            examGenerating={examGenerating}
            examActionBusy={examActionBusy}
            activeFocusSession={activeFocusSession}
            focusBusy={focusBusy}
            onSelectExam={setActiveExamId}
            onCreateExam={handleCreateExam}
            onStartExam={handleStartExam}
            onSubmitAnswer={handleSubmitAnswer}
            onFinishExam={handleFinishExam}
            onStartFocus={handleStartFocus}
            onStopFocus={handleStopFocus}
            showToast={showToast}
          />
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

      <FocusAlertSettingsModal
        show={isFocusAlertSettingsOpen}
        settings={focusAlertSettings}
        loading={focusAlertSettingsLoading}
        saving={focusAlertSettingsSaving}
        onClose={() => setIsFocusAlertSettingsOpen(false)}
        onPreview={playFocusAlertPreviewSound}
        onSave={handleSaveFocusAlertSettings}
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
                <X size={16} aria-hidden="true" />
              </button>
            </div>

            <div className="chat-search-input-wrap">
              <Search size={16} aria-hidden="true" />
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
