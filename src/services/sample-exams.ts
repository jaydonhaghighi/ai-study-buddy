/**
 * Sample exam configurations for testing
 */

import { ExamConfiguration } from '../types';

export const SAMPLE_EXAMS: ExamConfiguration[] = [
  {
    id: 'math-101-midterm',
    title: 'Pre-Calculus Midterm Exam',
    description: 'A comprehensive exam covering algebra, trigonometry, and basic calculus concepts.',
    totalTimeMs: 120 * 60 * 1000, // 120 minutes
    adaptiveDifficulty: true,
    sections: [
      {
        id: 'section-1',
        title: 'Algebra & Functions',
        description: 'Algebraic equations, functions, and transformations',
        timePerQuestionMs: 120000, // 2 minutes per question
        totalQuestionsTarget: 10,
        questions: [
          {
            id: 'q1',
            topic: 'Quadratic Equations',
            text: 'Solve for x: x² + 5x + 6 = 0',
            options: ['x = -2, -3', 'x = 2, 3', 'x = -2, 3', 'x = 2, -3'],
            correctAnswer: 0,
            initialDifficulty: 'easy',
            explanation: 'Factoring: (x+2)(x+3) = 0, so x = -2 or x = -3',
          },
          {
            id: 'q2',
            topic: 'Functions',
            text: 'If f(x) = 2x² - 3x + 1, find f(-2)',
            options: ['11', '15', '19', '23'],
            correctAnswer: 2,
            initialDifficulty: 'easy',
            explanation: 'f(-2) = 2(-2)² - 3(-2) + 1 = 8 + 6 + 1 = 15',
          },
          {
            id: 'q3',
            topic: 'Domain and Range',
            text: 'What is the domain of f(x) = √(x-3)?',
            options: ['All real numbers', 'x ≥ 3', 'x > 3', 'x ≤ 3'],
            correctAnswer: 1,
            initialDifficulty: 'medium',
            explanation: 'The radicand must be non-negative, so x - 3 ≥ 0, thus x ≥ 3',
          },
          {
            id: 'q4',
            topic: 'Polynomial Division',
            text: 'Divide x³ + 2x² - 5x - 6 by (x - 2)',
            options: ['x² + 4x + 3', 'x² + 3x - 4', 'x² - 4x + 3', 'x² + 5x + 5'],
            correctAnswer: 0,
            initialDifficulty: 'medium',
            explanation: 'Using synthetic division: quotient is x² + 4x + 3',
          },
          {
            id: 'q5',
            topic: 'Composite Functions',
            text: 'If f(x) = x + 2 and g(x) = x², find (f ∘ g)(3)',
            options: ['11', '13', '15', '17'],
            correctAnswer: 1,
            initialDifficulty: 'medium',
            explanation: '(f ∘ g)(3) = f(g(3)) = f(9) = 9 + 2 = 11',
          },
          {
            id: 'q6',
            topic: 'Quadratic Equations',
            text: 'Find the vertex of f(x) = x² - 4x + 3',
            options: ['(2, -1)', '(1, -2)', '(2, -3)', '(-2, 3)'],
            correctAnswer: 0,
            initialDifficulty: 'hard',
            explanation: 'Vertex form: h = -b/2a = 4/2 = 2; f(2) = 4 - 8 + 3 = -1',
          },
        ],
      },
      {
        id: 'section-2',
        title: 'Trigonometry',
        description: 'Trigonometric functions and identities',
        timePerQuestionMs: 120000,
        totalQuestionsTarget: 8,
        questions: [
          {
            id: 'trig1',
            topic: 'Basic Trigonometry',
            text: 'In a right triangle, if sin(θ) = 3/5, what is cos(θ)?',
            options: ['3/5', '4/5', '5/4', '5/3'],
            correctAnswer: 1,
            initialDifficulty: 'easy',
            explanation: 'Using the Pythagorean identity: sin²(θ) + cos²(θ) = 1; cos²(θ) = 1 - 9/25 = 16/25; cos(θ) = 4/5',
          },
          {
            id: 'trig2',
            topic: 'Trigonometric Identities',
            text: 'Simplify: sin(x)·csc(x)',
            options: ['sin(x)', '1', 'tan(x)', 'cos(x)'],
            correctAnswer: 1,
            initialDifficulty: 'medium',
            explanation: 'csc(x) = 1/sin(x), so sin(x)·csc(x) = sin(x)·(1/sin(x)) = 1',
          },
        ],
      },
    ],
  },
  {
    id: 'english-201-essay',
    title: 'English Literature Quiz',
    description: 'Multiple choice questions on classic English literature',
    totalTimeMs: 60 * 60 * 1000, // 60 minutes
    adaptiveDifficulty: true,
    sections: [
      {
        id: 'lit-section-1',
        title: 'Classic Literature',
        description: 'Questions on classic literature and authors',
        timePerQuestionMs: 120000,
        totalQuestionsTarget: 10,
        questions: [
          {
            id: 'lit1',
            topic: 'Shakespeare',
            text: 'Who is the protagonist of "Hamlet"?',
            options: ['Claudius', 'Gertrude', 'Prince Hamlet', 'Ophelia'],
            correctAnswer: 2,
            initialDifficulty: 'easy',
            explanation: 'Prince Hamlet is the main character and protagonist of the play.',
          },
          {
            id: 'lit2',
            topic: 'Themes in Literature',
            text: 'Which theme is central to "The Great Gatsby"?',
            options: ['Revenge', 'The American Dream', 'Time Travel', 'Family Secrets'],
            correctAnswer: 1,
            initialDifficulty: 'medium',
            explanation: 'The pursuit and corruption of the American Dream is the central theme.',
          },
          {
            id: 'lit3',
            topic: 'Historical Context',
            text: 'When was "Jane Eyre" by Charlotte Brontë published?',
            options: ['1817', '1847', '1877', '1897'],
            correctAnswer: 1,
            initialDifficulty: 'medium',
            explanation: '"Jane Eyre" was published in 1847 under the pen name Currer Bell.',
          },
        ],
      },
    ],
  },
];

export function getSampleExam(examId: string): ExamConfiguration | undefined {
  return SAMPLE_EXAMS.find((exam) => exam.id === examId);
}

export function getAllSampleExams(): ExamConfiguration[] {
  return SAMPLE_EXAMS;
}
