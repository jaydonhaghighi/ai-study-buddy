import { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, SendHorizontal } from 'lucide-react';

type SpeechRecognitionAlternativeLike = {
  transcript?: string;
};

type SpeechRecognitionResultLike = {
  isFinal?: boolean;
  length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
};

type SpeechRecognitionEventLike = Event & {
  results: {
    length: number;
    [index: number]: SpeechRecognitionResultLike;
  };
  error?: string;
};

type SpeechRecognitionLike = EventTarget & {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

type ChatInputProps = {
  selectedChatId: string | null;
  input: string;
  loading: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
};

export default function ChatInput({
  selectedChatId,
  input,
  loading,
  onInputChange,
  onSubmit,
}: ChatInputProps) {
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const baseInputRef = useRef('');
  const [speechSupported, setSpeechSupported] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const speechWindow = window as Window & {
      SpeechRecognition?: SpeechRecognitionCtor;
      webkitSpeechRecognition?: SpeechRecognitionCtor;
    };

    const RecognitionCtor = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!RecognitionCtor) {
      setSpeechSupported(false);
      return;
    }

    setSpeechSupported(true);
    const recognition = new RecognitionCtor();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      let finalTranscript = '';
      let interimTranscript = '';
      for (let i = 0; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result?.[0]?.transcript ?? '';
        if (!text) continue;
        if (result.isFinal) {
          finalTranscript += text;
        } else {
          interimTranscript += text;
        }
      }

      const combined = `${baseInputRef.current}${finalTranscript || interimTranscript}`.trim();
      onInputChange(combined);
    };

    recognition.onerror = (event: SpeechRecognitionEventLike) => {
      const code = event.error ?? 'unknown';
      if (code === 'not-allowed') {
        setSpeechError('Microphone permission was denied.');
      } else if (code === 'no-speech') {
        setSpeechError('No speech detected. Try again.');
      } else if (code === 'network') {
        setSpeechError('Speech service is unavailable right now.');
      } else {
        setSpeechError('Speech recognition failed.');
      }
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
      recognitionRef.current = null;
    };
  }, [onInputChange]);

  useEffect(() => {
    if (!loading) return;
    if (!isListening) return;
    recognitionRef.current?.stop();
  }, [isListening, loading]);

  const handleToggleSpeech = () => {
    if (!speechSupported || loading) return;
    const recognition = recognitionRef.current;
    if (!recognition) return;

    setSpeechError(null);
    if (isListening) {
      recognition.stop();
      setIsListening(false);
      return;
    }

    baseInputRef.current = input.trim().length > 0 ? `${input.trim()} ` : '';
    try {
      recognition.start();
      setIsListening(true);
    } catch {
      setSpeechError('Unable to start speech recognition.');
      setIsListening(false);
    }
  };

  const handleSubmit = (event: React.FormEvent) => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    }
    onSubmit(event);
  };

  if (!selectedChatId) return null;

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <div className="chat-input-wrapper">
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          disabled={loading}
        />
        <button
          type="button"
          className={`chat-mic-button ${isListening ? 'listening' : ''}`}
          onClick={handleToggleSpeech}
          disabled={loading || !speechSupported}
          title={
            !speechSupported
              ? 'Speech-to-text is not supported in this browser'
              : isListening
                ? 'Stop speech-to-text'
                : 'Start speech-to-text'
          }
          aria-label={isListening ? 'Stop speech-to-text' : 'Start speech-to-text'}
        >
          {isListening ? <MicOff size={18} aria-hidden="true" /> : <Mic size={18} aria-hidden="true" />}
          <span>{isListening ? 'Stop' : 'Speak'}</span>
        </button>
        <button type="submit" className="chat-send-button" disabled={!input.trim() || loading}>
          <SendHorizontal size={18} aria-hidden="true" />
          <span>Send</span>
        </button>
      </div>
      {(isListening || speechError) && (
        <div className={`chat-input-status ${speechError ? 'chat-input-status-error' : ''}`}>
          {speechError ?? 'Listening...'}
        </div>
      )}
    </form>
  );
}
