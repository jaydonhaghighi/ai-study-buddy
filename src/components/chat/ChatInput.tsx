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
  if (!selectedChatId) return null;

  return (
    <form className="chat-input-form" onSubmit={onSubmit}>
      <div className="chat-input-wrapper">
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="chat-send-button" disabled={!input.trim() || loading}>
          Send
        </button>
      </div>
    </form>
  );
}
