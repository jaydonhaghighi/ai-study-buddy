import { useEffect, useMemo, useState } from 'react';
import type { FlashcardReviewRating, StudySet, StudySource } from '../../types';
import type { StudySetPreset } from '../../services/study-set-service';

type ChatStudySetsPanelProps = {
  selectedChatId: string | null;
  studySets: StudySet[];
  activeStudySet: StudySet | null;
  activeStudySetId: string | null;
  studySetGenerating: boolean;
  reviewBusyCardId: string | null;
  onSelectStudySet: (studySetId: string) => void;
  onGenerateStudySet: (preset: StudySetPreset) => void;
  onReviewFlashcard: (studySetId: string, cardId: string, rating: FlashcardReviewRating) => void;
};

type StudyTab = 'quiz' | 'flashcards' | 'exam';

function formatTimestamp(date: Date | null): string {
  if (!date) return '-';
  return date.toLocaleString();
}

function sourceLookup(sources: StudySource[]): Record<string, StudySource> {
  const out: Record<string, StudySource> = {};
  for (const source of sources) {
    out[source.id] = source;
  }
  return out;
}

export default function ChatStudySetsPanel({
  selectedChatId,
  studySets,
  activeStudySet,
  activeStudySetId,
  studySetGenerating,
  reviewBusyCardId,
  onSelectStudySet,
  onGenerateStudySet,
  onReviewFlashcard,
}: ChatStudySetsPanelProps) {
  const [activeTab, setActiveTab] = useState<StudyTab>('quiz');
  const [revealAnswers, setRevealAnswers] = useState<Record<string, boolean>>({});
  const [flashcardIndex, setFlashcardIndex] = useState(0);
  const [flashcardFlipped, setFlashcardFlipped] = useState(false);

  useEffect(() => {
    setRevealAnswers({});
    setFlashcardIndex(0);
    setFlashcardFlipped(false);
  }, [activeStudySetId]);

  const sourceById = useMemo(
    () => sourceLookup(activeStudySet?.sources ?? []),
    [activeStudySet?.sources],
  );

  if (!selectedChatId) {
    return (
      <div className="preview-sidebar-empty">
        <p>Select a chat first, then generate quizzes and flashcards from your study history.</p>
      </div>
    );
  }

  return (
    <div className="study-set-panel">
      <div className="study-set-actions">
        <button
          type="button"
          className="study-set-generate-btn"
          onClick={() => onGenerateStudySet('quick')}
          disabled={studySetGenerating}
        >
          Quick
        </button>
        <button
          type="button"
          className="study-set-generate-btn"
          onClick={() => onGenerateStudySet('standard')}
          disabled={studySetGenerating}
        >
          Standard
        </button>
        <button
          type="button"
          className="study-set-generate-btn"
          onClick={() => onGenerateStudySet('exam')}
          disabled={studySetGenerating}
        >
          Exam Prep
        </button>
      </div>

      {studySetGenerating && <div className="study-set-status">Generating your study set...</div>}

      {studySets.length === 0 ? (
        <div className="study-set-empty">No study set yet. Generate one from this chat.</div>
      ) : (
        <>
          <div className="study-set-selector-row">
            <label htmlFor="study-set-selector">Study Set</label>
            <select
              id="study-set-selector"
              className="study-set-selector"
              value={activeStudySetId ?? (studySets[0]?.id ?? '')}
              onChange={(event) => onSelectStudySet(event.target.value)}
            >
              {studySets.map((studySet, index) => (
                <option key={studySet.id} value={studySet.id}>
                  {index === 0 ? 'Latest' : `Set ${studySets.length - index}`} • {formatTimestamp(studySet.createdAt)}
                </option>
              ))}
            </select>
          </div>

          {activeStudySet && activeStudySet.status === 'failed' && activeStudySet.errorMessage && (
            <div className="study-set-error">{activeStudySet.errorMessage}</div>
          )}

          {activeStudySet && activeStudySet.status === 'generating' && (
            <div className="study-set-status">This set is still generating...</div>
          )}

          {activeStudySet && activeStudySet.status === 'ready' && (
            <>
              <div className="study-set-tabs">
                <button
                  type="button"
                  className={`study-set-tab-btn ${activeTab === 'quiz' ? 'active' : ''}`}
                  onClick={() => setActiveTab('quiz')}
                >
                  Quiz
                </button>
                <button
                  type="button"
                  className={`study-set-tab-btn ${activeTab === 'flashcards' ? 'active' : ''}`}
                  onClick={() => setActiveTab('flashcards')}
                >
                  Flashcards
                </button>
                <button
                  type="button"
                  className={`study-set-tab-btn ${activeTab === 'exam' ? 'active' : ''}`}
                  onClick={() => setActiveTab('exam')}
                >
                  Exam
                </button>
              </div>

              {activeTab === 'quiz' && (
                <div className="study-set-list">
                  {activeStudySet.quizQuestions.map((question) => (
                    <div key={question.id} className="study-item-card">
                      <div className="study-item-header">
                        <span>{question.id}</span>
                        <span>{question.difficulty}</span>
                      </div>
                      <div className="study-item-title">{question.prompt}</div>
                      {question.questionType === 'mcq' && question.options.length > 0 && (
                        <ol className="study-item-options">
                          {question.options.map((option, idx) => (
                            <li key={`${question.id}-opt-${idx}`}>{option}</li>
                          ))}
                        </ol>
                      )}
                      <button
                        type="button"
                        className="study-item-action"
                        onClick={() => {
                          setRevealAnswers((prev) => ({
                            ...prev,
                            [question.id]: !prev[question.id],
                          }));
                        }}
                      >
                        {revealAnswers[question.id] ? 'Hide Answer' : 'Reveal Answer'}
                      </button>
                      {revealAnswers[question.id] && (
                        <div className="study-item-answer">
                          <strong>Answer:</strong> {question.correctAnswer}
                          {question.explanation && (
                            <p>{question.explanation}</p>
                          )}
                        </div>
                      )}
                      {question.sourceIds.length > 0 && (
                        <div className="study-source-chips">
                          {question.sourceIds.map((sourceId) => {
                            const source = sourceById[sourceId];
                            return (
                              <span key={`${question.id}-${sourceId}`} className="study-source-chip" title={source?.snippet || sourceId}>
                                {sourceId}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'flashcards' && activeStudySet.flashcards.length > 0 && (
                <div className="study-flashcard-panel">
                  <div className="study-flashcard-header">
                    <span>
                      Card {flashcardIndex + 1} / {activeStudySet.flashcards.length}
                    </span>
                    <div className="study-flashcard-nav">
                      <button
                        type="button"
                        className="study-item-action"
                        onClick={() => {
                          setFlashcardIndex((prev) => Math.max(0, prev - 1));
                          setFlashcardFlipped(false);
                        }}
                        disabled={flashcardIndex === 0}
                      >
                        Prev
                      </button>
                      <button
                        type="button"
                        className="study-item-action"
                        onClick={() => {
                          setFlashcardIndex((prev) => Math.min(activeStudySet.flashcards.length - 1, prev + 1));
                          setFlashcardFlipped(false);
                        }}
                        disabled={flashcardIndex === activeStudySet.flashcards.length - 1}
                      >
                        Next
                      </button>
                    </div>
                  </div>

                  {(() => {
                    const card = activeStudySet.flashcards[flashcardIndex];
                    if (!card) return null;
                    const busyId = `${activeStudySet.id}:${card.id}`;
                    const isBusy = reviewBusyCardId === busyId;
                    return (
                      <div className="study-item-card">
                        <div className="study-item-header">
                          <span>{card.id}</span>
                          <span>{card.difficulty}</span>
                        </div>
                        <div className="study-item-title">{flashcardFlipped ? card.back : card.front}</div>
                        <button
                          type="button"
                          className="study-item-action"
                          onClick={() => setFlashcardFlipped((value) => !value)}
                        >
                          {flashcardFlipped ? 'Show Front' : 'Flip Card'}
                        </button>
                        <div className="study-flashcard-meta">
                          Next review: {formatTimestamp(card.nextReviewAt)}
                        </div>
                        <div className="study-rate-row">
                          {(['again', 'hard', 'good', 'easy'] as FlashcardReviewRating[]).map((rating) => (
                            <button
                              key={`${card.id}-${rating}`}
                              type="button"
                              className="study-rate-btn"
                              onClick={() => onReviewFlashcard(activeStudySet.id, card.id, rating)}
                              disabled={isBusy}
                            >
                              {rating}
                            </button>
                          ))}
                        </div>
                        {card.sourceIds.length > 0 && (
                          <div className="study-source-chips">
                            {card.sourceIds.map((sourceId) => {
                              const source = sourceById[sourceId];
                              return (
                                <span key={`${card.id}-${sourceId}`} className="study-source-chip" title={source?.snippet || sourceId}>
                                  {sourceId}
                                </span>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}

              {activeTab === 'flashcards' && activeStudySet.flashcards.length === 0 && (
                <div className="study-set-empty">No flashcards in this set.</div>
              )}

              {activeTab === 'exam' && (
                <div className="study-set-list">
                  {activeStudySet.examQuestions.map((question) => (
                    <div key={question.id} className="study-item-card">
                      <div className="study-item-header">
                        <span>{question.id}</span>
                        <span>{question.difficulty}</span>
                      </div>
                      <div className="study-item-title">{question.prompt}</div>
                      {question.rubric.length > 0 && (
                        <ul className="study-item-rubric">
                          {question.rubric.map((criterion, idx) => (
                            <li key={`${question.id}-criterion-${idx}`}>{criterion}</li>
                          ))}
                        </ul>
                      )}
                      <details className="study-answer-details">
                        <summary>Show model answer</summary>
                        <p>{question.modelAnswer}</p>
                      </details>
                      {question.sourceIds.length > 0 && (
                        <div className="study-source-chips">
                          {question.sourceIds.map((sourceId) => {
                            const source = sourceById[sourceId];
                            return (
                              <span key={`${question.id}-${sourceId}`} className="study-source-chip" title={source?.snippet || sourceId}>
                                {sourceId}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
