/**
 * Exam Launcher - Shows available exams and launches them
 */

import { useEffect, useState } from 'react';
import { ExamConfiguration, ExamQuestion, QuizQuestion, StudyExamQuestion, StudySet } from '../../types';
import { getAllSampleExams } from '../../services/sample-exams';
import {
  ExamAttemptSummary,
  subscribeRecentExamReports,
} from '../../services/exam-report-service';
import { ChatSession } from './useChatCollections';
import './ExamLauncher.css';

interface ExamLauncherProps {
  userId: string;
  selectedChatId: string | null;
  onSelectChat: (chatId: string) => void;
  chats: ChatSession[];
  activeStudySet: StudySet | null;
  studySets: StudySet[];
  onSelectExam: (exam: ExamConfiguration) => void;
}

type CustomExamMode = 'mcq' | 'short' | 'mixed';

function inferTopic(question: QuizQuestion): string {
  const match = question.prompt.match(/^([^:?.!]{3,48})[:?.!]/);
  if (match && match[1]) {
    return match[1].trim();
  }
  return 'Chat Review';
}

function mapQuizToExamQuestion(question: QuizQuestion): ExamQuestion | null {
  if (question.options.length < 2) {
    return null;
  }

  let correctIndex = question.correctOptionIndex;
  if (correctIndex === null || correctIndex < 0 || correctIndex >= question.options.length) {
    const byTextIndex = question.options.findIndex((option) => option.trim() === question.correctAnswer.trim());
    correctIndex = byTextIndex >= 0 ? byTextIndex : null;
  }

  if (correctIndex === null) {
    return null;
  }

  return {
    id: `study-${question.id}`,
    topic: inferTopic(question),
    text: question.prompt,
    options: question.options,
    correctAnswer: correctIndex,
    questionType: 'mcq',
    initialDifficulty: question.difficulty,
    explanation: question.explanation || 'Review this concept from your study materials.',
  };
}

function mapStudyExamToQuestion(question: StudyExamQuestion): ExamQuestion {
  return {
    id: `study-open-${question.id}`,
    topic: 'Open Response',
    text: question.prompt,
    options: [],
    correctAnswer: 0,
    questionType: 'short',
    modelAnswer: question.modelAnswer,
    rubric: question.rubric,
    initialDifficulty: question.difficulty,
    explanation: 'Compare your response to the key points and model answer.',
  };
}

function buildStudySetExam(
  studySet: StudySet,
  desiredQuestionCount: number,
  mode: CustomExamMode,
  adaptiveDifficulty: boolean = true,
  fixedDifficulty?: 'easy' | 'medium' | 'hard',
): ExamConfiguration | null {
  const mcqQuestions = studySet.quizQuestions
    .map(mapQuizToExamQuestion)
    .filter((question): question is ExamQuestion => question !== null);
  const shortQuestions = studySet.examQuestions.map(mapStudyExamToQuestion);

  let examQuestions: ExamQuestion[] = [];
  if (mode === 'mcq') {
    examQuestions = mcqQuestions;
  } else if (mode === 'short') {
    examQuestions = shortQuestions;
  } else {
    examQuestions = [...mcqQuestions, ...shortQuestions];
  }

  if (examQuestions.length < 3) {
    return null;
  }

  // Filter by fixed difficulty if adaptive is disabled
  if (!adaptiveDifficulty && fixedDifficulty) {
    examQuestions = examQuestions.filter((q) => q.initialDifficulty === fixedDifficulty);
  }

  if (examQuestions.length < 3) {
    return null;
  }

  const questionCount = Math.min(desiredQuestionCount, examQuestions.length);
  const selectedQuestions = examQuestions.slice(0, questionCount);

  // When using fixed difficulty, all questions go into one section
  if (!adaptiveDifficulty && fixedDifficulty) {
    return {
      id: `study-set-exam-${studySet.id}-${mode}-${questionCount}-${fixedDifficulty}`,
      title: `Custom Exam (${mode === 'mixed' ? 'Mixed' : mode === 'short' ? 'Short Answer' : 'MCQ'}) - ${fixedDifficulty.charAt(0).toUpperCase() + fixedDifficulty.slice(1)}`,
      description: 'Built from your recent chat discussion and uploaded material analysis.',
      totalTimeMs: Math.max(15 * 60 * 1000, selectedQuestions.length * 90 * 1000),
      adaptiveDifficulty: false,
      sections: [{
        id: fixedDifficulty,
        title: `${fixedDifficulty.charAt(0).toUpperCase() + fixedDifficulty.slice(1)} Questions`,
        description: `All ${fixedDifficulty} level questions from your study set`,
        timePerQuestionMs: 90 * 1000,
        totalQuestionsTarget: selectedQuestions.length,
        questions: selectedQuestions,
      }],
    };
  }

  // Original logic for adaptive difficulty (progressive sections)
  const easy = selectedQuestions.filter((q) => q.initialDifficulty === 'easy');
  const medium = selectedQuestions.filter((q) => q.initialDifficulty === 'medium');
  const hard = selectedQuestions.filter((q) => q.initialDifficulty === 'hard');

  const orderedSections = [
    { id: 'easy', title: 'Warmup', description: 'Build confidence with fundamentals', questions: easy },
    { id: 'medium', title: 'Core Mastery', description: 'Apply key concepts from your chat and files', questions: medium },
    { id: 'hard', title: 'Stretch', description: 'Challenge-level practice for retention', questions: hard },
  ].filter((section) => section.questions.length > 0);

  if (orderedSections.length === 0) {
    return null;
  }

  const totalQuestions = orderedSections.reduce((sum, section) => sum + section.questions.length, 0);

  return {
    id: `study-set-exam-${studySet.id}-${mode}-${questionCount}`,
    title: `Custom Exam (${mode === 'mixed' ? 'Mixed' : mode === 'short' ? 'Short Answer' : 'MCQ'})`,
    description: 'Built from your recent chat discussion and uploaded material analysis.',
    totalTimeMs: Math.max(15 * 60 * 1000, totalQuestions * 90 * 1000),
    adaptiveDifficulty,
    sections: orderedSections.map((section) => ({
      id: section.id,
      title: section.title,
      description: section.description,
      timePerQuestionMs: 90 * 1000,
      totalQuestionsTarget: section.questions.length,
      questions: section.questions,
    })),
  };
}

export default function ExamLauncher({
  userId,
  selectedChatId,
  onSelectChat,
  chats,
  activeStudySet,
  studySets,
  onSelectExam,
}: ExamLauncherProps) {
  const availableExams = getAllSampleExams();
  const [recentAttempts, setRecentAttempts] = useState<ExamAttemptSummary[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(true);
  const [selectedAttempt, setSelectedAttempt] = useState<ExamAttemptSummary | null>(null);
  const [customMode, setCustomMode] = useState<CustomExamMode>('mixed');
  const [customQuestionCount, setCustomQuestionCount] = useState(10);
  const [customAdaptiveDifficulty, setCustomAdaptiveDifficulty] = useState(true);
  const [customFixedDifficulty, setCustomFixedDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [selectedExamForConfig, setSelectedExamForConfig] = useState<string | null>(null);
  const [examAdaptiveOverride, setExamAdaptiveOverride] = useState<boolean | null>(null);
  const [localSelectedStudySetId, setLocalSelectedStudySetId] = useState<string | null>(null);
  const readyStudySets = studySets.filter((set) => set.status === 'ready');
  const chatReadyStudySets = selectedChatId
    ? readyStudySets.filter((set) => set.chatId === selectedChatId)
    : [];
  const preferredStudySet = chatReadyStudySets.find((set) => set.id === localSelectedStudySetId)
    ?? (activeStudySet && activeStudySet.status === 'ready' && activeStudySet.chatId === selectedChatId ? activeStudySet : (chatReadyStudySets[0] ?? null));

  const buildRecoveryExam = (attempt: ExamAttemptSummary): ExamConfiguration | null => {
    const baseExam = availableExams.find((exam) => exam.id === attempt.examId);
    if (!baseExam || attempt.weakTopics.length === 0) return null;

    const weakTopicSet = new Set(attempt.weakTopics.map((t) => t.toLowerCase()));
    const filteredSections = baseExam.sections
      .map((section) => {
        const filteredQuestions = section.questions.filter((question) =>
          weakTopicSet.has(question.topic.toLowerCase())
        );

        return {
          ...section,
          questions: filteredQuestions,
          totalQuestionsTarget: filteredQuestions.length,
        };
      })
      .filter((section) => section.questions.length > 0);

    if (filteredSections.length === 0) return null;

    return {
      ...baseExam,
      id: `${baseExam.id}-recovery-${attempt.id}`,
      title: `${baseExam.title} - Recovery Drill`,
      description: `Targeted practice for weak areas: ${attempt.weakTopics.join(', ')}`,
      totalTimeMs: Math.max(10 * 60 * 1000, Math.round(baseExam.totalTimeMs * 0.5)),
      sections: filteredSections,
      adaptiveDifficulty: false,
    };
  };

  const buildRetryExam = (attempt: ExamAttemptSummary): ExamConfiguration | null => {
    const baseExam = availableExams.find((exam) => exam.id === attempt.examId);
    if (!baseExam) return null;

    return {
      ...baseExam,
      id: `${baseExam.id}-retry-${attempt.id}`,
      title: `${baseExam.title} - Retry`,
      description: `Retry attempt based on your exam from ${attempt.completedAt.toLocaleDateString()}`,
    };
  };

  useEffect(() => {
    // Auto-select first ready study set for this chat
    if (selectedChatId) {
      const chatStudySets = studySets.filter((set) => set.chatId === selectedChatId && set.status === 'ready');
      if (chatStudySets.length > 0) {
        setLocalSelectedStudySetId(chatStudySets[0].id);
      } else {
        setLocalSelectedStudySetId(null);
      }
    } else {
      setLocalSelectedStudySetId(null);
    }
  }, [selectedChatId, studySets]);

  useEffect(() => {
    setIsHistoryLoading(true);
    const unsubscribe = subscribeRecentExamReports(
      userId,
      (items) => {
        setRecentAttempts(items);
        setHistoryError(null);
        setIsHistoryLoading(false);
      },
      (error) => {
        setHistoryError(error.message || 'Could not load exam history.');
        setIsHistoryLoading(false);
      }
    );

    return () => unsubscribe();
  }, [userId]);

  const availableCustomQuestions = (() => {
    if (!preferredStudySet) return 0;
    let count = 0;
    if (customMode === 'mcq') {
      count = preferredStudySet.quizQuestions.length;
    } else if (customMode === 'short') {
      count = preferredStudySet.examQuestions.length;
    } else {
      count = preferredStudySet.quizQuestions.length + preferredStudySet.examQuestions.length;
    }

    if (customAdaptiveDifficulty) {
      return count;
    }

    let fixedCount = 0;
    if (customMode === 'mcq' || customMode === 'mixed') {
      fixedCount += preferredStudySet.quizQuestions.filter((q) => q.difficulty === customFixedDifficulty).length;
    }
    if (customMode === 'short' || customMode === 'mixed') {
      fixedCount += preferredStudySet.examQuestions.filter((q) => q.difficulty === customFixedDifficulty).length;
    }
    return fixedCount;
  })();

  return (
    <div className="exam-launcher">
      <div className="launcher-header">
        <h3>📝 Exam Simulations</h3>
        <p>Practice with timed mock exams to prepare for real assessments</p>
      </div>

      <section className="custom-exam-panel">
        <div>
          <h4>Generate Exam From Chat And Files</h4>
          <p>
            {selectedChatId
              ? 'Uses your study set from the selected chat, including uploaded material context.'
              : 'Select a chat first, then generate a study set in the Study panel to unlock this option.'}
          </p>
          {selectedChatId && chatReadyStudySets.length === 0 && (
            <p className="custom-hint">
              No ready study set found for this chat yet. Open Study panel and generate one first.
            </p>
          )}
          <div className="custom-exam-controls">
            <label className="custom-control">
              <span>Chat</span>
              <select
                value={selectedChatId || ''}
                onChange={(event) => onSelectChat(event.target.value)}
              >
                <option value="">— Select a chat —</option>
                {chats.map((chat) => (
                  <option key={chat.id} value={chat.id}>
                    {chat.name}
                  </option>
                ))}
              </select>
            </label>
            {selectedChatId && chatReadyStudySets.length > 0 && (
              <label className="custom-control">
                <span>Study Set</span>
                <select
                  value={localSelectedStudySetId || ''}
                  onChange={(event) => setLocalSelectedStudySetId(event.target.value || null)}
                >
                  <option value="">— Select a study set —</option>
                  {chatReadyStudySets
                    .map((set, idx, arr) => (
                      <option key={set.id} value={set.id}>
                        {idx === 0 ? 'Latest' : `Set ${arr.length - idx}`} • {set.createdAt ? new Date(set.createdAt).toLocaleDateString() : 'Unknown'}
                      </option>
                    ))}
                </select>
              </label>
            )}
            <label className="custom-control">
              <span>Mode</span>
              <select
                value={customMode}
                onChange={(event) => setCustomMode(event.target.value as CustomExamMode)}
              >
                <option value="mixed">Mixed</option>
                <option value="mcq">MCQ Only</option>
                <option value="short">Short Answer Only</option>
              </select>
            </label>
            <label className="custom-control">
              <span>Questions</span>
              <select
                value={customQuestionCount}
                onChange={(event) => setCustomQuestionCount(Number(event.target.value))}
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
              </select>
            </label>
            <label className="custom-control checkbox-control">
              <input
                type="checkbox"
                checked={customAdaptiveDifficulty}
                onChange={(event) => setCustomAdaptiveDifficulty(event.target.checked)}
              />
              <span>{customAdaptiveDifficulty ? '🔄 Adaptive' : '📊 Fixed'} Difficulty</span>
            </label>
            {!customAdaptiveDifficulty && (
              <label className="custom-control">
                <span>Level</span>
                <select
                  value={customFixedDifficulty}
                  onChange={(event) => setCustomFixedDifficulty(event.target.value as 'easy' | 'medium' | 'hard')}
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </label>
            )}
            <span className="custom-availability">Available: {availableCustomQuestions}</span>
          </div>
        </div>
        <button
          className="btn-start-exam"
          onClick={() => {
            if (!preferredStudySet) return;
            const customExam = buildStudySetExam(
              preferredStudySet,
              customQuestionCount,
              customMode,
              customAdaptiveDifficulty,
              !customAdaptiveDifficulty ? customFixedDifficulty : undefined
            );
            if (customExam) {
              onSelectExam(customExam);
            }
          }}
          disabled={!selectedChatId || !preferredStudySet || availableCustomQuestions < 3}
          title={!selectedChatId
            ? 'Select a chat first'
            : !preferredStudySet
              ? 'Select or generate a study set for the selected chat'
              : availableCustomQuestions < 3
                ? `Not enough ${customAdaptiveDifficulty ? 'questions' : `${customFixedDifficulty} questions`} available in this mode`
              : ''}
        >
          Start Custom Exam
        </button>
      </section>

      <div className="exams-grid">
        {availableExams.map((exam) => (
          <div key={exam.id} className="exam-card">
            <div className="exam-card-header">
              <h4>{exam.title}</h4>
              <span className="exam-difficulty">
                {exam.adaptiveDifficulty ? '🔄 Adaptive' : '📊 Fixed'}
              </span>
            </div>
            <p className="exam-description">{exam.description}</p>
            <div className="exam-info">
              <div className="info-item">
                <span className="label">Duration</span>
                <span className="value">{Math.round(exam.totalTimeMs / 60000)} min</span>
              </div>
              <div className="info-item">
                <span className="label">Questions</span>
                <span className="value">
                  {exam.sections.reduce((sum, s) => sum + s.totalQuestionsTarget, 0)}
                </span>
              </div>
              <div className="info-item">
                <span className="label">Sections</span>
                <span className="value">{exam.sections.length}</span>
              </div>
            </div>
            <button 
              className="btn-start-exam"
              onClick={() => setSelectedExamForConfig(exam.id)}
            >
              Start Exam →
            </button>
          </div>
        ))}
      </div>

      {selectedExamForConfig && (
        <div className="exam-config-modal">
          <div className="exam-config-content">
            <h4>Exam Configuration</h4>
            <p>Customize the difficulty setting before starting:</p>
            <div className="config-option">
              <label className="checkbox-control">
                <input
                  type="checkbox"
                  checked={examAdaptiveOverride !== false}
                  onChange={() => setExamAdaptiveOverride(examAdaptiveOverride === true ? false : true)}
                />
                <span>{examAdaptiveOverride !== false ? '🔄 Adaptive' : '📊 Fixed'} Difficulty</span>
              </label>
              <p className="config-description">
                {examAdaptiveOverride !== false
                  ? 'Questions adjust based on your performance. Getting answers right makes questions harder, missing makes them easier.'
                  : 'All questions stay at their original difficulty level throughout the exam.'}
              </p>
            </div>
            <div className="config-actions">
              <button
                className="btn-primary"
                onClick={() => {
                  const selectedExam = availableExams.find((e) => e.id === selectedExamForConfig);
                  if (selectedExam && examAdaptiveOverride !== null) {
                    const modifiedExam = {
                      ...selectedExam,
                      adaptiveDifficulty: examAdaptiveOverride,
                      id: `${selectedExam.id}-${examAdaptiveOverride ? 'adaptive' : 'fixed'}`,
                    };
                    onSelectExam(modifiedExam);
                    setSelectedExamForConfig(null);
                    setExamAdaptiveOverride(null);
                  }
                }}
              >
                Start Exam
              </button>
              <button
                className="btn-secondary"
                onClick={() => {
                  setSelectedExamForConfig(null);
                  setExamAdaptiveOverride(null);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="exam-features">
        <h4>Features of Exam Simulation Mode:</h4>
        <ul>
          <li>⏰ Countdown timer with real-time tracking</li>
          <li>🎯 Adaptive difficulty based on your performance</li>
          <li>😊 Confidence tracking for each answer</li>
          <li>📊 Detailed performance breakdown by topic</li>
          <li>⚠️ Weak topic identification</li>
          <li>⏱️ Time loss analysis and insights</li>
          <li>🎯 Personalized recovery plan</li>
        </ul>
      </div>

      <div className="exam-history">
        <h4>Recent Attempts</h4>
        {isHistoryLoading ? (
          <p className="history-empty">Loading recent exam attempts...</p>
        ) : historyError ? (
          <p className="history-error">{historyError}</p>
        ) : recentAttempts.length === 0 ? (
          <p className="history-empty">No completed attempts yet. Start your first exam.</p>
        ) : (
          <div className="history-list">
            {recentAttempts.map((attempt) => (
              <article key={attempt.id} className="history-item">
                <div className="history-item-header">
                  <h5>{attempt.examTitle}</h5>
                  <span className="history-score">{attempt.scorePercent}%</span>
                </div>
                <p>
                  {attempt.correctAnswers}/{attempt.totalQuestions} correct • {Math.round(attempt.totalDurationMs / 60000)} min
                </p>
                <p className="history-date">{attempt.completedAt.toLocaleString()}</p>
                {attempt.weakTopics.length > 0 && (
                  <p className="history-weak-topics">Weak topics: {attempt.weakTopics.join(', ')}</p>
                )}
                <div className="history-actions">
                  <button
                    className="history-action-btn tertiary"
                    onClick={() => {
                      const retryExam = buildRetryExam(attempt);
                      if (retryExam) {
                        onSelectExam(retryExam);
                      }
                    }}
                    disabled={!availableExams.some((exam) => exam.id === attempt.examId)}
                  >
                    Retry Full Exam
                  </button>
                  <button
                    className="history-action-btn"
                    onClick={() => setSelectedAttempt(attempt)}
                  >
                    Review Report
                  </button>
                  <button
                    className="history-action-btn secondary"
                    onClick={() => {
                      const recoveryExam = buildRecoveryExam(attempt);
                      if (recoveryExam) {
                        onSelectExam(recoveryExam);
                      }
                    }}
                    disabled={attempt.weakTopics.length === 0}
                    title={attempt.weakTopics.length === 0 ? 'No weak topics found for this attempt' : ''}
                  >
                    Practice Weak Topics
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      {selectedAttempt && (
        <div className="exam-report-overlay" role="dialog" aria-modal="true" onClick={() => setSelectedAttempt(null)}>
          <div className="exam-report-modal" onClick={(e) => e.stopPropagation()}>
            <div className="exam-report-modal-header">
              <h4>{selectedAttempt.examTitle}</h4>
              <button className="modal-close" onClick={() => setSelectedAttempt(null)}>
                ✕
              </button>
            </div>
            {selectedAttempt.report ? (
              <>
                <p>
                  Score: {Math.round(selectedAttempt.report.performanceMetrics.accuracyPercent)}% ({selectedAttempt.report.performanceMetrics.correctAnswers}/{selectedAttempt.report.performanceMetrics.totalQuestions})
                </p>
                <p>
                  Avg time/question: {Math.round(selectedAttempt.report.performanceMetrics.averageTimePerQuestion / 1000)}s
                </p>
                {selectedAttempt.report.topicPerformance.length > 0 && (
                  <div className="modal-section">
                    <h5>Topic Performance</h5>
                    <ul>
                      {selectedAttempt.report.topicPerformance.map((topic) => (
                        <li key={`${selectedAttempt.id}-${topic.topic}`}>
                          {topic.topic}: {Math.round(topic.accuracyPercent)}% ({topic.correctAnswers}/{topic.questionsAttempted})
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedAttempt.report.recoveryPlan.length > 0 && (
                  <div className="modal-section">
                    <h5>Recovery Plan</h5>
                    <ul>
                      {selectedAttempt.report.recoveryPlan.map((item, idx) => (
                        <li key={`${selectedAttempt.id}-plan-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p>Detailed report data was not available for this attempt.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
