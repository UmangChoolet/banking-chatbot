import { useState, useRef, useEffect, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import ChatWindow from "./components/ChatWindow";
import Sidebar from "./components/Sidebar";
import UploadModal from "./components/UploadModal";
import { useChat } from "./hooks/useChat";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export default function App() {
  const [sessionId] = useState(() => uuidv4());
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [documents, setDocuments] = useState([]);

  const { messages, isLoading, sendMessage, clearHistory } = useChat({
    sessionId,
    apiBase: API_BASE,
  });

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (e) {
      console.error("Failed to fetch documents", e);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUploadComplete = () => {
    setUploadOpen(false);
    fetchDocuments();
  };

  const handleClear = async () => {
    try {
      await fetch(`${API_BASE}/chat/${sessionId}`, { method: "DELETE" });
    } catch (e) {}
    clearHistory();
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            <span />
            <span />
            <span />
          </button>
          <div className="brand">
            <div className="brand-icon">⬡</div>
            <div className="brand-text">
              <span className="brand-name">FinBot</span>
              <span className="brand-sub">AI Banking Assistant</span>
            </div>
          </div>
        </div>
        <div className="header-right">
          <div className="status-pill">
            <span className="status-dot" />
            Online
          </div>
          <button
            className="btn-upload"
            onClick={() => setUploadOpen(true)}
          >
            + Upload Doc
          </button>
          <button className="btn-clear" onClick={handleClear} title="Clear chat">
            ↺ New Chat
          </button>
        </div>
      </header>

      <div className="app-body">
        <Sidebar
          open={sidebarOpen}
          documents={documents}
          onClose={() => setSidebarOpen(false)}
          onUpload={() => setUploadOpen(true)}
        />

        <main className="main-content">
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            onSend={sendMessage}
            sessionId={sessionId}
          />
        </main>
      </div>

      {uploadOpen && (
        <UploadModal
          apiBase={API_BASE}
          onClose={() => setUploadOpen(false)}
          onComplete={handleUploadComplete}
        />
      )}
    </div>
  );
}
