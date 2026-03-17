import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import type { Citation } from '../../types';
import logo from '../../public/logo.png';

type MessageLike = {
  id: string;
  text: string;
  userName: string;
  isAI?: boolean;
  model?: string;
  citations?: Citation[];
};

type ChatMessageListProps = {
  selectedChatId: string | null;
  visibleMessages: MessageLike[];
  loading: boolean;
  hasStreamingAiText: boolean;
  messagesContainerRef: React.RefObject<HTMLDivElement>;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  onMessagesScroll: () => void;
};

export default function ChatMessageList({
  selectedChatId,
  visibleMessages,
  loading,
  hasStreamingAiText,
  messagesContainerRef,
  messagesEndRef,
  onMessagesScroll,
}: ChatMessageListProps) {
  const formatAiName = (name: string | undefined) => {
    if (!name || name === 'AI Study Buddy') return 'Echelon';
    return name;
  };

  return (
    <div className="chat-messages" ref={messagesContainerRef} onScroll={onMessagesScroll}>
      {!selectedChatId ? (
        <div className="chat-welcome">
          <div className="welcome-icon" aria-hidden="true">
            <img src={logo} alt="" className="welcome-logo" />
          </div>
          <h3 className="welcome-title">Welcome to Echelon</h3>
          <p className="welcome-message">To get started, choose an existing chat or create a new one from the sidebar</p>
        </div>
      ) : (
        <>
          {visibleMessages.map((message) => (
            <div key={message.id} className={`message ${!message.isAI ? 'message-user' : 'message-ai'}`}>
              <div className="message-content">
                <div className="message-header">
                  <span className="message-name">{!message.isAI ? 'You' : formatAiName(message.userName)}</span>
                  {message.model && message.isAI && <span className="message-model">{message.model}</span>}
                </div>
                <div className="message-text">
                  {!message.isAI ? (
                    <div className="plain-text">{message.text}</div>
                  ) : (
                    <div className="markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                          a: ({ children, ...props }) => (
                            <a {...props} target="_blank" rel="noopener noreferrer">
                              {children}
                            </a>
                          ),
                          code: ({ children, className, ...props }) => {
                            const isBlock = !!className && className.includes('language-');
                            return isBlock ? (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            ) : (
                              <code className="inline-code" {...props}>
                                {children}
                              </code>
                            );
                          },
                        }}
                      >
                        {message.text}
                      </ReactMarkdown>

                      {Array.isArray(message.citations) && message.citations.length > 0 && (
                        <div className="message-citations">
                          <div className="message-citations-title">Sources</div>
                          {message.citations.map((citation) => (
                            <div key={`${message.id}-${citation.id}`} className="message-citation-item">
                              <div className="message-citation-label">
                                [{citation.id}] {citation.fileName} ({citation.locationLabel})
                              </div>
                              <div className="message-citation-snippet">{citation.snippet}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {loading && !hasStreamingAiText && (
            <div className="message message-ai">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
}
