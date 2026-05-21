import { useRef, useEffect, useState } from "react";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";

const SUGGESTED_QUESTIONS = [
  "What are the personal loan interest rates?",
  "How do I apply for a credit card?",
  "What documents do I need for a home loan?",
  "What are the FD interest rates for senior citizens?",
  "How do I dispute a credit card transaction?",
  "What is the minimum balance for a savings account?",
];

export default function ChatWindow({ messages, isLoading, onSend, sessionId }) {
  const bottomRef = useRef(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (messages.length > 1) setShowSuggestions(false);
  }, [messages.length]);

  const handleSuggestion = (q) => {
    setShowSuggestions(false);
    onSend(q);
  };

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && messages[messages.length - 1]?.loading !== true && (
          <div className="typing-indicator">
            <span />
            <span />
            <span />
          </div>
        )}

        {showSuggestions && messages.length <= 1 && (
          <div className="suggestions-grid">
            <p className="suggestions-label">Try asking:</p>
            <div className="suggestions">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  className="suggestion-chip"
                  onClick={() => handleSuggestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={onSend} disabled={isLoading} />
    </div>
  );
}
