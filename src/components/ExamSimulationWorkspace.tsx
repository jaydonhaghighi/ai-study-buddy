import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  ExamConfidence,
  ExamSimulation,
  ExamCompletionReason,
} from '../types';
import type {
  FocusSession,
  FocusStartOptions,
  FocusStopResult,
} from './chat/useFocusTracking';
import './ExamSimulationWorkspace.css';

type ToastVariant = 'success' | 'warning' | 'info';

type CurrentChatLike = {
  id: string;
  name: string;
  courseId: string;
  sessionId: string;
};

type ExamSimulationWorkspaceProps = {
  selectedChatId: string | null;
  currentChat: CurrentChatLike | null;
  indexedMaterialsCount: number;
  selectedModel: string;
  examSimulations: ExamSimulation[];
  activeExam: ExamSimulation | null;
  activeExamId: string | null;
  examGenerating: boolean;
  examActionBusy: boolean;
  activeFocusSession: FocusSession | null;
  focusBusy: boolean;
  onSelectExam: (examId: string) => void;
  onCreateExam: () => Promise<boolean>;
  onStartExam: (examId: string) => Promise<boolean>;
  onSubmitAnswer: (params: {
    examSimulationId: string;
    questionId: string;
    selectedOptionIndex: number;
    confidence: ExamConfidence;
    elapsedSec: number;
  }) => Promise<boolean>;
  onFinishExam: (params: {
    examSimulationId: string;
    completionReason: ExamCompletionReason;
    focusSessionId?: string;
  }) => Promise<boolean>;
  onStartFocus: (options?: FocusStartOptions) => void;
  onStopFocus: () => Promise<FocusStopResult | null>;
  showToast: (message: string, variant?: ToastVariant) => void;
};

function formatClock(totalSec: number): string {
  const seconds = Math.max(0, Math.floor(totalSec));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function formatTimestamp(date: Date | null): string {
  if (!date) return '-';
  return date.toLocaleString();
}

function formatStatus(status: ExamSimulation['status']): string {
  if (status === 'in_progress') return 'In Progress';
  if (status === 'timed_out') return 'Timed Out';
  if (status === 'abandoned') return 'Abandoned';
  if (status === 'failed') return 'Failed';
  if (status === 'completed') return 'Completed';
  if (status === 'generating') return 'Generating';
  return 'Ready';
}

export default function ExamSimulationWorkspace({
  selectedChatId,
  currentChat,
  indexedMaterialsCount,
  selectedModel,
  examSimulations,
  activeExam,
  activeExamId,
  examGenerating,
  examActionBusy,
  activeFocusSession,
  focusBusy,
  onSelectExam,
  onCreateExam,
  onStartExam,
  onSubmitAnswer,
  onFinishExam,
  onStartFocus,
  onStopFocus,
  showToast,
}: ExamSimulationWorkspaceProps) {
  const [selectedOptionIndex, setSelectedOptionIndex] = useState<number | null>(null);
  const [selectedConfidence, setSelectedConfidence] = useState<ExamConfidence | null>(null);
  const [focusEnabled, setFocusEnabled] = useState(false);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const finalizingRef = useRef(false);

  const currentQuestion = useMemo(
    () => activeExam?.servedQuestions.find((question) => question.id === activeExam.currentQuestionId) ?? null,
    [activeExam],
  );

  const isExamFocusLinked =
    activeFocusSession?.mode === 'exam_simulation' &&
    activeFocusSession.examSimulationId != null &&
    activeFocusSession.examSimulationId === activeExam?.id;

  useEffect(() => {
    setSelectedOptionIndex(null);
    setSelectedConfidence(null);
  }, [activeExam?.id, activeExam?.currentQuestionId]);

  useEffect(() => {
    if (activeExam?.status !== 'in_progress') return;
    const tick = () => setNowMs(Date.now());
    tick();
    const timerId = window.setInterval(tick, 250);
    return () => window.clearInterval(timerId);
  }, [activeExam?.status]);

  const remainingSec = useMemo(() => {
    if (activeExam?.status !== 'in_progress' || !activeExam.endsAt) return activeExam?.durationSec ?? 0;
    return Math.max(0, Math.ceil((activeExam.endsAt.getTime() - nowMs) / 1000));
  }, [activeExam?.durationSec, activeExam?.endsAt, activeExam?.status, nowMs]);

  const answeredCount = activeExam?.responses.length ?? 0;
  const currentQuestionNumber = currentQuestion ? answeredCount + 1 : answeredCount;

  const finalizeExam = useCallback(async (completionReason: ExamCompletionReason) => {
    if (!activeExam || finalizingRef.current) return;
    finalizingRef.current = true;

    let focusSessionId: string | undefined;
    if (isExamFocusLinked) {
      const stopResult = await onStopFocus();
      focusSessionId = stopResult?.ok ? stopResult.focusSessionId : undefined;
    }

    await onFinishExam({
      examSimulationId: activeExam.id,
      completionReason,
      focusSessionId,
    });
    finalizingRef.current = false;
  }, [activeExam, isExamFocusLinked, onFinishExam, onStopFocus]);

  useEffect(() => {
    if (!activeExam || activeExam.status !== 'in_progress') return;
    if (remainingSec > 0) return;
    if (finalizingRef.current) return;
    void finalizeExam('time_up');
  }, [activeExam, finalizeExam, remainingSec]);

  useEffect(() => {
    if (!activeExam || activeExam.status !== 'in_progress') return;
    if (activeExam.currentQuestionId) return;
    if (answeredCount === 0) return;
    if (finalizingRef.current) return;
    void finalizeExam('submitted');
  }, [activeExam, answeredCount, finalizeExam]);

  const handleStartClick = async () => {
    if (!activeExam || activeExam.status !== 'ready' || !currentChat) return;

    if (!focusEnabled) {
      await onStartExam(activeExam.id);
      return;
    }

    if (activeFocusSession && !isExamFocusLinked) {
      showToast('Stop your current focus session before starting a mock exam with focus tracking.', 'warning');
      return;
    }

    onStartFocus({
      courseId: currentChat.courseId,
      sessionId: currentChat.sessionId,
      mode: 'exam_simulation',
      chatId: currentChat.id,
      examSimulationId: activeExam.id,
      onStarted: async () => {
        const started = await onStartExam(activeExam.id);
        if (!started) {
          window.setTimeout(() => {
            void onStopFocus();
          }, 350);
        }
      },
    });
  };

  const handleSubmitAnswerClick = async () => {
    if (!activeExam || !currentQuestion || selectedOptionIndex == null || !selectedConfidence) return;
    const ok = await onSubmitAnswer({
      examSimulationId: activeExam.id,
      questionId: currentQuestion.id,
      selectedOptionIndex,
      confidence: selectedConfidence,
      elapsedSec: Math.max(0, activeExam.durationSec - remainingSec),
    });
    if (ok) {
      setSelectedOptionIndex(null);
      setSelectedConfidence(null);
    }
  };

  const handleExitClick = () => {
    if (!activeExam) return;
    const confirmed = window.confirm('Exit this mock exam and mark it as abandoned?');
    if (!confirmed) return;
    void finalizeExam('abandoned');
  };

  const sectionProgress = useMemo(() => {
    if (!activeExam) return [];
    return activeExam.sections.map((section, index) => {
      const answeredInSection = activeExam.responses.filter((response) => response.sectionId === section.id).length;
      const isCurrent = currentQuestion?.sectionId === section.id;
      const isComplete = answeredInSection >= section.questionTargetCount;
      return {
        ...section,
        answeredInSection,
        status: isComplete ? 'complete' : isCurrent ? 'current' : index === 0 && answeredInSection === 0 ? 'current' : 'upcoming',
      };
    });
  }, [activeExam, currentQuestion?.sectionId]);

  if (!selectedChatId || !currentChat) {
    return (
      <div className="exam-workspace exam-workspace-empty">
        <div className="exam-empty-card">
          <h3>Select a chat to build a mock exam</h3>
          <p>Live Exam Simulation Mode uses the active chat and its indexed materials as the exam source.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`exam-workspace ${activeExam?.status === 'in_progress' ? 'exam-workspace-live' : ''}`}>
      <section className="exam-hero-card">
        <div>
          <p className="exam-eyebrow">Live Exam Simulation</p>
          <h2>{currentChat.name}</h2>
          <p className="exam-hero-copy">
            Standard Mock uses the current chat, the selected model, and any indexed materials to build a timed adaptive exam.
          </p>
        </div>
        <div className="exam-hero-meta">
          <div className="exam-hero-stat">
            <span>Model</span>
            <strong>{selectedModel}</strong>
          </div>
          <div className="exam-hero-stat">
            <span>Indexed materials</span>
            <strong>{indexedMaterialsCount}</strong>
          </div>
          <div className="exam-hero-stat">
            <span>Preset</span>
            <strong>30 min / 10 Q</strong>
          </div>
        </div>
      </section>

      <section className="exam-shell-card">
        <div className="exam-shell-header">
          <div>
            <h3>Mock Exam Runs</h3>
            <p>Generate a fresh run or resume the latest one for this chat.</p>
          </div>
          <div className="exam-shell-actions">
            {examSimulations.length > 0 && (
              <select
                className="exam-select"
                value={activeExamId ?? examSimulations[0].id}
                onChange={(event) => onSelectExam(event.target.value)}
                disabled={activeExam?.status === 'in_progress'}
              >
                {examSimulations.map((exam, index) => (
                  <option key={exam.id} value={exam.id}>
                    {index === 0 ? 'Latest' : `Run ${examSimulations.length - index}`} • {formatStatus(exam.status)} • {formatTimestamp(exam.createdAt)}
                  </option>
                ))}
              </select>
            )}
            <button
              type="button"
              className="exam-btn exam-btn-secondary"
              onClick={() => void onCreateExam()}
              disabled={examGenerating || examActionBusy || activeExam?.status === 'in_progress'}
            >
              {examGenerating ? 'Generating...' : 'Generate New Mock'}
            </button>
          </div>
        </div>

        {!activeExam && (
          <div className="exam-empty-state">
            <p>No mock exam yet. Generate one to create a grounded 30-minute practice run.</p>
          </div>
        )}

        {activeExam?.status === 'generating' && (
          <div className="exam-status-panel">
            <h4>Generating your exam bank...</h4>
            <p>We’re building 15 grounded questions across easy, medium, and hard difficulty.</p>
          </div>
        )}

        {activeExam?.status === 'failed' && (
          <div className="exam-status-panel exam-status-panel-error">
            <h4>Mock exam generation failed</h4>
            <p>{activeExam.errorMessage || 'Try adding more study context and generating again.'}</p>
          </div>
        )}

        {activeExam?.status === 'ready' && (
          <div className="exam-ready-grid">
            <div className="exam-ready-card">
              <h4>Ready to Start</h4>
              <p>One preset only in v1: 30 minutes, 10 served questions, 3 pacing sections, adaptive difficulty.</p>
              <div className="exam-preset-list">
                <div className="exam-preset-item"><span>Total time</span><strong>30 min</strong></div>
                <div className="exam-preset-item"><span>Question bank</span><strong>15 MCQs</strong></div>
                <div className="exam-preset-item"><span>Served live</span><strong>10 questions</strong></div>
                <div className="exam-preset-item"><span>Difficulty</span><strong>Adaptive</strong></div>
              </div>
            </div>

            <div className="exam-ready-card">
              <h4>Launch Options</h4>
              <label className="exam-toggle-row">
                <input
                  type="checkbox"
                  checked={focusEnabled}
                  onChange={(event) => setFocusEnabled(event.target.checked)}
                  disabled={focusBusy || (activeFocusSession != null && !isExamFocusLinked)}
                />
                <span>Track focus during the mock exam</span>
              </label>
              <p className="exam-muted">
                Focus tracking is optional and only affects the recap, not the difficulty changes.
              </p>
              <button
                type="button"
                className="exam-btn exam-btn-primary"
                onClick={() => void handleStartClick()}
                disabled={examActionBusy || examGenerating || focusBusy}
              >
                Start Mock Exam
              </button>
            </div>
          </div>
        )}

        {activeExam?.status === 'in_progress' && currentQuestion && (
          <div className="exam-live-layout">
            <div className="exam-live-header">
              <div>
                <p className="exam-eyebrow">No-distraction mode</p>
                <h3>Question {currentQuestionNumber} of {activeExam.servedQuestionCount}</h3>
              </div>
              <div className="exam-live-header-right">
                <div className="exam-timer-chip">{formatClock(remainingSec)}</div>
                {isExamFocusLinked && (
                  <div className="exam-focus-chip">Focus tracking on</div>
                )}
              </div>
            </div>

            <div className="exam-section-row">
              {sectionProgress.map((section) => (
                <div
                  key={section.id}
                  className={`exam-section-chip exam-section-chip-${section.status}`}
                >
                  <span>{section.label}</span>
                  <strong>{section.answeredInSection}/{section.questionTargetCount}</strong>
                </div>
              ))}
            </div>

            <div className="exam-question-card">
              <div className="exam-question-meta">
                <span>{currentQuestion.topic}</span>
                <span>{currentQuestion.difficulty}</span>
              </div>
              <h4>{currentQuestion.prompt}</h4>
              <div className="exam-option-list">
                {currentQuestion.options.map((option, index) => (
                  <button
                    key={`${currentQuestion.id}-${index}`}
                    type="button"
                    className={`exam-option-btn ${selectedOptionIndex === index ? 'selected' : ''}`}
                    onClick={() => setSelectedOptionIndex(index)}
                    disabled={examActionBusy}
                  >
                    <span className="exam-option-label">{String.fromCharCode(65 + index)}</span>
                    <span>{option}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="exam-confidence-card">
              <h5>How confident are you?</h5>
              <div className="exam-confidence-row">
                {(['low', 'medium', 'high'] as ExamConfidence[]).map((confidence) => (
                  <button
                    key={confidence}
                    type="button"
                    className={`exam-confidence-btn ${selectedConfidence === confidence ? 'selected' : ''}`}
                    onClick={() => setSelectedConfidence(confidence)}
                    disabled={examActionBusy}
                  >
                    {confidence}
                  </button>
                ))}
              </div>
            </div>

            <div className="exam-live-actions">
              <button
                type="button"
                className="exam-btn exam-btn-secondary"
                onClick={handleExitClick}
                disabled={examActionBusy}
              >
                Exit Exam
              </button>
              <button
                type="button"
                className="exam-btn exam-btn-secondary"
                onClick={() => void finalizeExam('submitted')}
                disabled={examActionBusy}
              >
                Submit Exam
              </button>
              <button
                type="button"
                className="exam-btn exam-btn-primary"
                onClick={() => void handleSubmitAnswerClick()}
                disabled={examActionBusy || selectedOptionIndex == null || selectedConfidence == null}
              >
                Submit Answer
              </button>
            </div>
          </div>
        )}

        {activeExam?.status === 'in_progress' && !currentQuestion && (
          <div className="exam-status-panel">
            <h4>Finalizing your mock exam...</h4>
            <p>We’re turning your answers into a performance breakdown and recovery plan.</p>
          </div>
        )}

        {activeExam && activeExam.status !== 'ready' && activeExam.status !== 'generating' && activeExam.status !== 'failed' && activeExam.status !== 'in_progress' && (
          <div className="exam-results-grid">
            <div className="exam-results-card">
              <div className="exam-results-head">
                <div>
                  <p className="exam-eyebrow">{formatStatus(activeExam.status)}</p>
                  <h4>Performance Breakdown</h4>
                </div>
                <div className="exam-score-pill">
                  {Math.round(activeExam.recap?.scorePercent ?? 0)}%
                </div>
              </div>
              <div className="exam-summary-stats">
                <div><span>Answered</span><strong>{activeExam.recap?.answeredCount ?? 0}/{activeExam.recap?.totalQuestionCount ?? activeExam.servedQuestionCount}</strong></div>
                <div><span>Correct</span><strong>{activeExam.recap?.correctCount ?? 0}</strong></div>
                <div><span>Overconfidence misses</span><strong>{activeExam.recap?.overconfidenceMisses ?? 0}</strong></div>
              </div>
              <p className="exam-summary-copy">
                {activeExam.recap?.weakTopicSummary || 'This run is ready for review.'}
              </p>
            </div>

            <div className="exam-results-card">
              <h4>Weak Topics</h4>
              {activeExam.recap?.weakTopics.length ? (
                <ul className="exam-list">
                  {activeExam.recap.weakTopics.map((topic) => (
                    <li key={topic.topic}>
                      <span>{topic.topic}</span>
                      <strong>{Math.round(topic.accuracyPercent)}%</strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="exam-muted">No topic breakdown available for this run.</p>
              )}
            </div>

            <div className="exam-results-card">
              <h4>Time-Loss Moments</h4>
              {activeExam.recap?.timeLossMoments.length ? (
                <ul className="exam-list exam-list-text">
                  {activeExam.recap.timeLossMoments.map((moment, index) => (
                    <li key={`${activeExam.id}-moment-${index}`}>{moment}</li>
                  ))}
                </ul>
              ) : (
                <p className="exam-muted">No major pace losses were recorded in this run.</p>
              )}
            </div>

            <div className="exam-results-card">
              <h4>Recovery Plan</h4>
              {activeExam.recap?.recoveryPlan.length ? (
                <ol className="exam-ordered-list">
                  {activeExam.recap.recoveryPlan.map((step, index) => (
                    <li key={`${activeExam.id}-step-${index}`}>{step}</li>
                  ))}
                </ol>
              ) : (
                <p className="exam-muted">No recovery plan was generated for this run.</p>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
