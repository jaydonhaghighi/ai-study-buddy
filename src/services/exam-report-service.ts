import {
  addDoc,
  collection,
  limit,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  Timestamp,
  where,
} from 'firebase/firestore';
import { db } from '../firebase-config';
import type { ExamReport } from '../types';

export interface ExamAttemptSummary {
  id: string;
  userId: string;
  examId: string;
  examTitle: string;
  completedAt: Date;
  scorePercent: number;
  totalQuestions: number;
  correctAnswers: number;
  totalDurationMs: number;
  weakTopics: string[];
  report?: ExamReport;
}

function normalizeReport(rawReport: unknown): ExamReport | undefined {
  if (!rawReport || typeof rawReport !== 'object') return undefined;
  const reportRecord = rawReport as Record<string, unknown>;
  const completedAtRaw = reportRecord.completedAt;
  const completedAt =
    completedAtRaw instanceof Timestamp ? completedAtRaw.toDate() : new Date();

  return {
    ...(reportRecord as unknown as ExamReport),
    completedAt,
  };
}

export async function saveExamReport(
  userId: string,
  examId: string,
  report: ExamReport
): Promise<void> {
  await addDoc(collection(db, 'examReports'), {
    userId,
    examId,
    examTitle: report.examTitle,
    completedAt: Timestamp.fromDate(report.completedAt),
    scorePercent: Math.round(report.performanceMetrics.accuracyPercent),
    totalQuestions: report.performanceMetrics.totalQuestions,
    correctAnswers: report.performanceMetrics.correctAnswers,
    totalDurationMs: report.totalDurationMs,
    weakTopics: report.weakTopics,
    report,
    createdAt: serverTimestamp(),
  });
}

export function subscribeRecentExamReports(
  userId: string,
  onData: (items: ExamAttemptSummary[]) => void,
  onError?: (error: Error) => void
): () => void {
  const q = query(
    collection(db, 'examReports'),
    where('userId', '==', userId),
    orderBy('createdAt', 'desc'),
    limit(8)
  );

  return onSnapshot(
    q,
    (snapshot) => {
      const items: ExamAttemptSummary[] = snapshot.docs.map((docSnapshot) => {
        const data = docSnapshot.data();
        const completedAtRaw = data.completedAt;
        const completedAt =
          completedAtRaw instanceof Timestamp ? completedAtRaw.toDate() : new Date();

        return {
          id: docSnapshot.id,
          userId: String(data.userId ?? ''),
          examId: String(data.examId ?? ''),
          examTitle: String(data.examTitle ?? 'Untitled Exam'),
          completedAt,
          scorePercent: Number(data.scorePercent ?? 0),
          totalQuestions: Number(data.totalQuestions ?? 0),
          correctAnswers: Number(data.correctAnswers ?? 0),
          totalDurationMs: Number(data.totalDurationMs ?? 0),
          weakTopics: Array.isArray(data.weakTopics) ? data.weakTopics.map(String) : [],
          report: normalizeReport(data.report),
        };
      });
      onData(items);
    },
    (error) => {
      if (onError) {
        onError(error);
      }
    }
  );
}
