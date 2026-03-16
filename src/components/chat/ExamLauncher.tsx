/**
 * Exam Launcher - Shows available exams and launches them
 */

import { useEffect, useState } from 'react';
import { ExamConfiguration } from '../../types';
import { getAllSampleExams } from '../../services/sample-exams';
import {
  ExamAttemptSummary,
  subscribeRecentExamReports,
} from '../../services/exam-report-service';
import './ExamLauncher.css';

interface ExamLauncherProps {
  userId: string;
  onSelectExam: (exam: ExamConfiguration) => void;
}

export default function ExamLauncher({ userId, onSelectExam }: ExamLauncherProps) {
  const availableExams = getAllSampleExams();
  const [recentAttempts, setRecentAttempts] = useState<ExamAttemptSummary[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(true);
  const [selectedAttempt, setSelectedAttempt] = useState<ExamAttemptSummary | null>(null);

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

  return (
    <div className="exam-launcher">
      <div className="launcher-header">
        <h3>📝 Exam Simulations</h3>
        <p>Practice with timed mock exams to prepare for real assessments</p>
      </div>

      <div className="exams-grid">
        {availableExams.map((exam) => (
          <div key={exam.id} className="exam-card" onClick={() => onSelectExam(exam)}>
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
            <button className="btn-start-exam">
              Start Exam →
            </button>
          </div>
        ))}
      </div>

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
