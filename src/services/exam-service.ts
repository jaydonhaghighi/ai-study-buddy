/**
 * Exam Simulation Service
 * Core logic for exam sessions, adaptive difficulty, and analysis
 */

import {
  ExamSession,
  AnsweredQuestion,
  PerformanceMetrics,
  TopicPerformance,
  ExamReport,
} from '../types';

/**
 * Calculate adaptive difficulty for next question based on performance
 */
export function getAdaptiveDifficulty(
  currentSession: ExamSession,
  adaptiveDifficultyEnabled: boolean
): 'easy' | 'medium' | 'hard' {
  if (!adaptiveDifficultyEnabled || currentSession.answeredQuestions.length === 0) {
    return 'medium';
  }

  const recent = currentSession.answeredQuestions.slice(-5);
  const correctCount = recent.filter((q) => q.isCorrect).length;
  const correctRatio = correctCount / recent.length;

  // If student is doing well, increase difficulty
  if (correctRatio >= 0.8) {
    return 'hard';
  }
  // If student is struggling, decrease difficulty
  if (correctRatio <= 0.4) {
    return 'easy';
  }
  return 'medium';
}

/**
 * Calculate performance metrics from answered questions
 */
export function calculatePerformanceMetrics(
  answeredQuestions: AnsweredQuestion[]
): PerformanceMetrics {
  const totalQuestions = answeredQuestions.length;
  const correctAnswers = answeredQuestions.filter((q) => q.isCorrect).length;
  const accuracyPercent = totalQuestions > 0 ? (correctAnswers / totalQuestions) * 100 : 0;

  const totalTimeMs = answeredQuestions.reduce((sum, q) => sum + q.timeSpentMs, 0);
  const averageTimePerQuestion = totalQuestions > 0 ? totalTimeMs / totalQuestions : 0;

  // Calculate time wasted (time spent on incorrect answers)
  const timeWastedMs = answeredQuestions
    .filter((q) => !q.isCorrect)
    .reduce((sum, q) => sum + q.timeSpentMs, 0);

  // Calculate confidence alignment (how well confidence predicted correctness)
  const confidenceCorrect = answeredQuestions
    .filter((q) => {
      if (q.confidence === 'high') return q.isCorrect;
      if (q.confidence === 'low') return !q.isCorrect;
      return true;
    }).length;
  const confidenceAlignment = totalQuestions > 0 ? (confidenceCorrect / totalQuestions) * 100 : 0;

  return {
    totalQuestions,
    correctAnswers,
    accuracyPercent,
    averageTimePerQuestion,
    timeWastedMs,
    confidenceAlignment,
  };
}

/**
 * Analyze performance by topic
 */
export function analyzeTopicPerformance(
  answeredQuestions: AnsweredQuestion[]
): TopicPerformance[] {
  const topicMap = new Map<string, AnsweredQuestion[]>();

  // Group questions by topic
  answeredQuestions.forEach((q) => {
    if (!topicMap.has(q.topic)) {
      topicMap.set(q.topic, []);
    }
    topicMap.get(q.topic)!.push(q);
  });

  // Calculate metrics for each topic
  return Array.from(topicMap.entries()).map(([topic, questions]) => {
    const correctAnswers = questions.filter((q) => q.isCorrect).length;
    const accuracy = (correctAnswers / questions.length) * 100;
    const avgTime = questions.reduce((sum, q) => sum + q.timeSpentMs, 0) / questions.length;
    const difficulty = questions[0].difficulty; // Use first question's difficulty as representative

    return {
      topic,
      questionsAttempted: questions.length,
      correctAnswers,
      accuracyPercent: accuracy,
      averageTimeMs: avgTime,
      difficulty,
    };
  });
}

/**
 * Generate recovery plan based on weak topics and time issues
 */
export function generateRecoveryPlan(
  topicPerformance: TopicPerformance[],
  performanceMetrics: PerformanceMetrics,
  sessionDurationMs: number
): string[] {
  const plan: string[] = [];

  // Identify weak topics (< 60% accuracy)
  const weakTopics = topicPerformance.filter((t) => t.accuracyPercent < 60);
  if (weakTopics.length > 0) {
    plan.push(
      `📚 Focus on mastering: ${weakTopics.map((t) => t.topic).join(', ')} (accuracy below 60%)`
    );
  }

  // Identify strong topics (> 80% accuracy)
  const strongTopics = topicPerformance.filter((t) => t.accuracyPercent > 80);
  if (strongTopics.length > 0) {
    plan.push(
      `✅ Maintain strength in: ${strongTopics.map((t) => t.topic).join(', ')} (excellent performance)`
    );
  }

  // Time management feedback
  if (performanceMetrics.timeWastedMs > sessionDurationMs * 0.2) {
    plan.push(
      `⏱️ Time management: You spent ${Math.round(performanceMetrics.timeWastedMs / 1000)}s on incorrect answers - consider moving on from difficult questions faster`
    );
  }

  if (performanceMetrics.averageTimePerQuestion > 60000) {
    // > 1 minute per question
    plan.push(
      `⚡ Speed up: Average ${Math.round(performanceMetrics.averageTimePerQuestion / 1000)}s per question - aim to reduce to 45-50s`
    );
  }

  // Confidence alignment feedback
  if (performanceMetrics.confidenceAlignment < 70) {
    plan.push(
      `🎯 Confidence calibration: Your confidence doesn't match your accuracy - reassess your knowledge of topics where you felt uncertain`
    );
  }

  // Difficulty progression feedback
  const hardQs = topicPerformance.filter((t) => t.difficulty === 'hard');
  if (hardQs.length > 0) {
    const hardAccuracy = hardQs.reduce((sum, t) => sum + t.accuracyPercent, 0) / hardQs.length;
    if (hardAccuracy < 50) {
      plan.push(`💪 Advanced concepts: Work on harder difficulty questions - target areas where you score below 50%`);
    }
  }

  if (plan.length === 0) {
    plan.push(`🌟 Excellent performance! Continue current study strategy.`);
  }

  return plan;
}

/**
 * Identify time loss moments (questions that took unusually long)
 */
export function identifyTimeLossAreas(
  answeredQuestions: AnsweredQuestion[]
): { moment: string; extraTimeMs: number }[] {
  if (answeredQuestions.length === 0) return [];

  const avgTime = answeredQuestions.reduce((sum, q) => sum + q.timeSpentMs, 0) / answeredQuestions.length;
  const threshold = avgTime * 1.5; // 50% above average

  return answeredQuestions
    .filter((q) => q.timeSpentMs > threshold)
    .map((q) => ({
      moment: `Question on "${q.topic}" (${q.timeSpentMs > 120000 ? 'spent 2+ min' : 'spent 1-2 min'})`,
      extraTimeMs: q.timeSpentMs - avgTime,
    }));
}

/**
 * Generate complete exam report
 */
export function generateExamReport(
  examSession: ExamSession,
  examTitle: string,
  totalDurationMs: number
): ExamReport {
  const topicPerformance = analyzeTopicPerformance(examSession.answeredQuestions);
  const performanceMetrics = calculatePerformanceMetrics(examSession.answeredQuestions);
  const recoveryPlan = generateRecoveryPlan(topicPerformance, performanceMetrics, totalDurationMs);
  const timeLossAreas = identifyTimeLossAreas(examSession.answeredQuestions);

  const weakTopics = topicPerformance
    .filter((t) => t.accuracyPercent < 60)
    .map((t) => t.topic);

  const strongTopics = topicPerformance
    .filter((t) => t.accuracyPercent > 80)
    .map((t) => t.topic);

  return {
    sessionId: examSession.id,
    examTitle,
    completedAt: examSession.completedAt || new Date(),
    totalDurationMs,
    performanceMetrics,
    topicPerformance,
    weakTopics,
    strongTopics,
    timeLossAreas: timeLossAreas.slice(0, 5), // Top 5 time loss areas
    recoveryPlan,
  };
}
