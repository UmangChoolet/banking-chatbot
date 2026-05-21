export default function Sidebar({ open, documents, onClose, onUpload }) {
  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onClose} />}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="sidebar-header">
          <h2>Knowledge Base</h2>
          <button className="sidebar-close" onClick={onClose}>✕</button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">How It Works</div>
          <div className="rag-steps">
            <div className="rag-step">
              <span className="step-num">1</span>
              <div>
                <strong>Upload</strong> banking documents (PDF, TXT, DOCX)
              </div>
            </div>
            <div className="rag-step">
              <span className="step-num">2</span>
              <div>
                <strong>Chunked</strong> into passages & embedded with AI
              </div>
            </div>
            <div className="rag-step">
              <span className="step-num">3</span>
              <div>
                <strong>Stored</strong> in FAISS vector database
              </div>
            </div>
            <div className="rag-step">
              <span className="step-num">4</span>
              <div>
                <strong>Retrieved</strong> by semantic similarity search
              </div>
            </div>
            <div className="rag-step">
              <span className="step-num">5</span>
              <div>
                <strong>LLM generates</strong> context-aware answers
              </div>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">
            Ingested Documents ({documents.length})
          </div>
          {documents.length === 0 ? (
            <p className="no-docs">
              No custom documents yet. Pre-loaded banking FAQs are active.
            </p>
          ) : (
            <div className="doc-list">
              {documents.map((doc, i) => (
                <div key={i} className="doc-item">
                  <span className="doc-icon">
                    {doc.file_type === ".pdf" ? "📕" : doc.file_type === ".docx" ? "📘" : "📄"}
                  </span>
                  <div className="doc-info">
                    <span className="doc-name">{doc.original_name}</span>
                    <span className="doc-meta">
                      {doc.chunks} chunks · {doc.size_mb}MB
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
          <button className="btn-upload-sidebar" onClick={onUpload}>
            + Upload Document
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Built-in Knowledge</div>
          <ul className="builtin-list">
            <li>📋 Personal Loans Guide</li>
            <li>💳 Credit Card Policies</li>
            <li>🏦 Banking FAQs</li>
            <li>📈 FD & Savings Rates</li>
            <li>🏠 Home Loan Info</li>
          </ul>
        </div>
      </aside>
    </>
  );
}
