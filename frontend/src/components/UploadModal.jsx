import { useState, useRef } from "react";

export default function UploadModal({ apiBase, onClose, onComplete }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    setError("");
    setResult(null);
    const allowed = [".pdf", ".txt", ".docx"];
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Allowed: PDF, TXT, DOCX`);
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      setError("File too large. Maximum size is 20MB.");
      return;
    }
    setFile(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${apiBase}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Upload failed. Check backend connection.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Upload Banking Document</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {!result ? (
            <>
              <div
                className={`drop-zone ${dragOver ? "drag-over" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept=".pdf,.txt,.docx"
                  style={{ display: "none" }}
                  onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                />
                {file ? (
                  <div className="file-selected">
                    <span className="file-icon">
                      {file.name.endsWith(".pdf") ? "📕" : file.name.endsWith(".docx") ? "📘" : "📄"}
                    </span>
                    <div>
                      <div className="file-name">{file.name}</div>
                      <div className="file-size">
                        {(file.size / 1024).toFixed(1)} KB
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="drop-prompt">
                    <div className="drop-icon">📤</div>
                    <p><strong>Drag & drop</strong> or click to browse</p>
                    <p className="drop-hint">PDF, TXT, DOCX · Max 20MB</p>
                  </div>
                )}
              </div>

              {error && <p className="upload-error">⚠️ {error}</p>}

              <div className="modal-info">
                <p>Uploaded documents are ingested into the RAG pipeline:</p>
                <ul>
                  <li>Text is extracted and chunked automatically</li>
                  <li>Embeddings are generated locally (free)</li>
                  <li>Stored in FAISS vector database</li>
                  <li>Available instantly for semantic search</li>
                </ul>
              </div>

              <div className="modal-actions">
                <button className="btn-cancel" onClick={onClose}>Cancel</button>
                <button
                  className={`btn-primary ${!file || uploading ? "disabled" : ""}`}
                  onClick={handleUpload}
                  disabled={!file || uploading}
                >
                  {uploading ? (
                    <>
                      <span className="spinner-sm" /> Processing...
                    </>
                  ) : (
                    "Ingest Document"
                  )}
                </button>
              </div>
            </>
          ) : (
            <div className="upload-success">
              <div className="success-icon">✅</div>
              <h3>Document Ingested Successfully!</h3>
              <div className="success-stats">
                <div className="stat">
                  <span className="stat-value">{result.chunks_created}</span>
                  <span className="stat-label">Chunks Created</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{result.total_vectors}</span>
                  <span className="stat-label">Total Vectors</span>
                </div>
              </div>
              <p className="success-msg">
                <strong>{result.file_name}</strong> is now part of the knowledge base.
                You can ask questions about its contents!
              </p>
              <button className="btn-primary" onClick={onComplete}>
                Start Chatting
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
