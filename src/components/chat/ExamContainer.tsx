import { useState } from 'react';
import { ExamSession, ExamConfiguration } from '../../types';
import { generateExamReport } from '../../services/exam-service';
import { saveExamReport } from '../../services/exam-report-service';
import { useExamSession } from './useExamSession';
import './ExamContainer.css';

interface ExamContainerProps {
  examConfig: ExamConfiguration;
  userId: string;
  onComplete: (report: ReturnType<typeof generateExamReport>) => void;
  onExit: () => void;
}

export default function ExamContainer({
  examConfig,
  userId,
  onComplete,
  onExit,
}: ExamContainerProps) {
  const { examSession, isActive, timeRemainingMs, startExam, submitAnswer, completeExam, getNextDifficulty, getCurrentQuestion } = useExamSession({
    examConfig,
    userId,
  });

  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<'low' | 'medium' | 'high'>('medium');
  const [showingReport, setShowingReport] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  const currentQuestion = getCurrentQuestion();
  const progress =
    examSession && currentQuestion
      ? `${examSession.answeredQuestions.length + 1}/${examConfig.sections.reduce((sum, s) => sum + s.totalQuestionsTarget, 0)}`
      : '0/0';

  const handleStartExam = () => {
    startExam();
    setHasStarted(true);
  };

  const handleSubmitAnswer = async () => {
    if (!currentQuestion || selectedAnswer === null || !examSession) return;

    submitAnswer(
      currentQuestion.id,
      selectedAnswer,
      currentQuestion.topic,
      currentQuestion.correctAnswer,
      confidence,
      currentQuestion.initialDifficulty
    );

    setSelectedAnswer(null);
    setConfidence('medium');
  };

  const handleCompleteExam = () => {
    const completedSession = completeExam();
    if (completedSession) {
      const report = generateExamReport(completedSession, examConfig.title, examConfig.totalTimeMs);
      setShowingReport(true);
      onComplete(report);
      void saveExamReport(userId, examConfig.id, report).catch((error: unknown) => {
        console.error('Failed to save exam report:', error);
      });
    }
  };

  const handleExit = () => {
    if (isActive && window.confirm('Are you sure you want to exit the exam? Your progress will be lost.')) {
      onExit();
    } else if (!isActive) {
      onExit();
    }
  };

  if (!hasStarted) {
    return (
      <div className="exam-start-modal">
        <div className="exam-start-content">
          <h2>{examConfig.title}</h2>
          <p className="exam-start-description">{examConfig.description}</p>
          <div className="exam-start-details">
            <div className="detail-item">
              <span className="detail-label">Total Time:</span>
              <span className="detail-value">{Math.round(examConfig.totalTimeMs / 60000)} minutes</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Sections:</span>
              <span className="detail-value">{examConfig.sections.length}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Total Questions:</span>
              <span className="detail-value">
                {examConfig.sections.reduce((sum, s) => sum + s.totalQuestionsTarget, 0)}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Adaptive Difficulty:</span>
              <span className="detail-value">{examConfig.adaptiveDifficulty ? 'Enabled' : 'Disabled'}</span>
            </div>
          </div>
          <div className="exam-start-buttons">
            <button onClick={handleStartExam} className="btn-primary btn-large">
              Start Exam
            </button>
            <button onClick={handleExit} className="btn-secondary btn-large">
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!currentQuestion || !examSession) {
    return (
      <div className="exam-container">
        <div className="exam-error">
          <h3>No questions available</h3>
          <button onClick={handleExit} className="btn-secondary">
            Exit Exam
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="exam-container">
      {showingReport ? (
        <ExamReport examSession={examSession} examConfig={examConfig} onExit={handleExit} />
      ) : (
        <>
          <div className="exam-header">
            <div className="exam-title">
              <h2>{examConfig.title}</h2>
            </div>
            <div className="exam-stats">
              <div className="stat progress">
                <span className="label">Progress</span>
                <span className="value">{progress}</span>
              </div>
              <div className="stat timer" style={{ color: timeRemainingMs < 60000 ? '#d32f2f' : 'inherit' }}>
                <span className="label">Time Remaining</span>
                <span className="value">{formatTime(timeRemainingMs)}</span>
              </div>
              <div className="stat difficulty">
                <span className="label">Difficulty</span>
                <span className="value difficulty-badge">{getNextDifficulty()}</span>
              </div>
            </div>
            <button onClick={handleExit} className="btn-exit" title="Exit exam">
              ✕
            </button>
          </div>

          <div className="exam-content">
            <div className="question-container">
              <div className="question-header">
                <span className="topic-badge">{currentQuestion.topic}</span>
                <span className="question-number">Question {examSession.currentQuestionIndex + 1}</span>
              </div>

              <h3 className="question-text">{currentQuestion.text}</h3>

              <div className="options-list">
                {currentQuestion.options.map((option, idx) => (
                  <label key={idx} className={`option-item ${selectedAnswer === idx ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="answer"
                      value={idx}
                      checked={selectedAnswer === idx}
                      onChange={() => setSelectedAnswer(idx)}
                      disabled={!isActive}
                    />
                    <span className="option-text">{option}</span>
                  </label>
                ))}
              </div>

              <div className="confidence-section">
                <label className="confidence-label">How confident are you?</label>
                <div className="confidence-options">
                  {(['low', 'medium', 'high'] as const).map((level) => (
                    <button
                      key={level}
                      onClick={() => setConfidence(level)}
                      className={`confidence-btn ${confidence === level ? 'active' : ''}`}
                    >
                      {level === 'low' ? '😕' : level === 'medium' ? '😐' : '😊'}
                      <span>{level}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="question-actions">
                {examSession.answeredQuestions.length < (examConfig.sections.reduce((sum, s) => sum + s.totalQuestionsTarget, 0) - 1) ? (
                  <>
                    <button
                      onClick={handleSubmitAnswer}
                      disabled={selectedAnswer === null}
                      className="btn-primary"
                    >
                      Next Question
                    </button>
                  </>
                ) : (
                  <>
                    <button onClick={handleSubmitAnswer} disabled={selectedAnswer === null} className="btn-primary">
                      Submit Last Answer
                    </button>
                    <button
                      onClick={() => {
                        handleSubmitAnswer();
                        setTimeout(handleCompleteExam, 100);
                      }}
                      className="btn-success"
                    >
                      Finish Exam
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

interface ExamReportProps {
  examSession: ExamSession;
  examConfig: ExamConfiguration;
  onExit: () => void;
}

function ExamReport({ examSession, examConfig, onExit }: ExamReportProps) {
  const report = generateExamReport(examSession, examConfig.title, examConfig.totalTimeMs);

  return (
    <div className="exam-report">
      <div className="report-header">
        <h2>Exam Complete!</h2>
        <p className="exam-title">{report.examTitle}</p>
      </div>

      <div className="report-metrics">
        <div className="metric-card primary">
          <span className="metric-label">Score</span>
          <span className="metric-value">{Math.round(report.performanceMetrics.accuracyPercent)}%</span>
          <span className="metric-subtitle">
            {report.performanceMetrics.correctAnswers}/{report.performanceMetrics.totalQuestions} Correct
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Time Spent</span>
          <span className="metric-value">{formatTime(report.totalDurationMs)}</span>
          <span className="metric-subtitle">Avg: {Math.round(report.performanceMetrics.averageTimePerQuestion / 1000)}s per Q</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Confidence Alignment</span>
          <span className="metric-value">{Math.round(report.performanceMetrics.confidenceAlignment)}%</span>
          <span className="metric-subtitle">Confidence vs Accuracy</span>
        </div>
      </div>

      <section className="report-section">
        <h3>📊 Topic Performance</h3>
        <div className="topic-grid">
          {report.topicPerformance.map((topic, idx) => (
            <div key={idx} className="topic-card">
              <h4>{topic.topic}</h4>
              <div className="topic-stat">
                <span>{topic.correctAnswers}/{topic.questionsAttempted}</span>
                <span className="accuracy">{Math.round(topic.accuracyPercent)}%</span>
              </div>
              <span className="difficulty-pill">{topic.difficulty}</span>
            </div>
          ))}
        </div>
      </section>

      {report.weakTopics.length > 0 && (
        <section className="report-section warning">
          <h3>⚠️ Weak Topics (Below 60% Accuracy)</h3>
          <ul className="topic-list">
            {report.weakTopics.map((topic, idx) => (
              <li key={idx}>{topic}</li>
            ))}
          </ul>
        </section>
      )}

      {report.timeLossAreas.length > 0 && (
        <section className="report-section">
          <h3>⏱️ Time Loss Moments</h3>
          <ul className="time-loss-list">
            {report.timeLossAreas.map((area, idx) => (
              <li key={idx}>
                <span>{area.moment}</span>
                <span className="time-value">+{Math.round(area.extraTimeMs / 1000)}s</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="report-section recovery">
        <h3>🎯 Recovery Plan</h3>
        <ul className="recovery-plan">
          {report.recoveryPlan.map((plan, idx) => (
            <li key={idx}>{plan}</li>
          ))}
        </ul>
      </section>

      <div className="report-actions">
        <button onClick={onExit} className="btn-primary btn-large">
          Close Report
        </button>
      </div>
    </div>
  );
}
