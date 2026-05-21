import { useState } from "react";

// Simple markdown renderer
function renderMarkdown(text) {
  if (!text) return "";
  
  const lines = text.split("\n");
  let html = "";
  let inList = false;

  for (let line of lines) {
    // Apply inline formatting
    line = line
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>");

    if (/^#{1,3} (.+)/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += line.replace(/^### (.+)/, "<h3>$1</h3>")
                  .replace(/^## (.+)/, "<h2>$1</h2>")
                  .replace(/^# (.+)/, "<h1>$1</h1>");
    } else if (/^\d+\.\s(.+)/.test(line) || /^[-•*]\s(.+)/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${line.replace(/^\d+\.\s/, "").replace(/^[-•*]\s/, "")}</li>`;
    } else if (line.trim() === "") {
      if (inList) { html += "</ul>"; inList = false; }
      html += "<br/>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${line}</p>`;
    }
  }

  if (inList) html += "</ul>";
  return html;
}

function formatTime(date) {
  return new Date(date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";
  const hasSources = message.sources && message.sources.length > 0;

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      {!isUser && (
        <div className="avatar assistant-avatar">⬡</div>
      )}

      <div className="message-content-wrap">
        <div
          className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} ${
            message.loading ? "bubble-loading" : ""
          } ${message.error ? "bubble-error" : ""}`}
        >
          {message.loading && !message.content ? (
            <div className="typing-dots">
              <span />
              <span />
              <span />
            </div>
          ) : (
            <div
              className="message-text"
              dangerouslySetInnerHTML={{
                __html: renderMarkdown(message.content),
              }}
            />
          )}
        </div>

        <div className="message-meta">
          <span className="message-time">{formatTime(message.timestamp)}</span>
          {hasSources && (
            <button
              className="sources-toggle"
              onClick={() => setShowSources(!showSources)}
            >
              {showSources ? "▲" : "▼"} {message.sources.length} source
              {message.sources.length > 1 ? "s" : ""}
            </button>
          )}
        </div>

        {showSources && hasSources && (
          <div className="sources-panel">
            <p className="sources-title">Retrieved Documents</p>
            {message.sources.map((src, i) => (
              <div key={i} className="source-item">
                <div className="source-header">
                  <span className="source-name">📄 {src.source}</span>
                  <span className="source-score">
                    {(src.relevance_score * 100).toFixed(0)}% match
                  </span>
                </div>
                <p className="source-excerpt">{src.excerpt}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <div className="avatar user-avatar">U</div>
      )}
    </div>
  );
}