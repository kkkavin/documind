import React, { useState, useEffect, useRef } from "react";
import { Settings, Upload, Send, FileText, Trash2, Bot, User, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import SettingsModal from "./components/SettingsModal";
import { processPdfFile } from "./services/pdf";
import { saveDocumentWithChunks, db } from "./services/db";
import { retrieveRelevantChunks, formatContext } from "./services/retrieval";
import { generateCompletionStream } from "./services/llm";

export default function App() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isProcessingPdf, setIsProcessingPdf] = useState(false);
  const [activeDocName, setActiveDocName] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const chatEndRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  // Safely load existing doc name on startup
  useEffect(() => {
    const loadCurrentDoc = async () => {
      try {
        const docs = await db.documents.toArray();
        if (docs && docs.length > 0) {
          setActiveDocName(docs[0].name);
        }
      } catch (err) {
        console.error("Failed to load existing document from IndexedDB:", err);
      }
    };
    loadCurrentDoc();
  }, []);

  // Handle PDF Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || file.type !== "application/pdf") {
      alert("Please upload a valid PDF file.");
      return;
    }

    setIsProcessingPdf(true);
    try {
      const chunks = await processPdfFile(file);
      await saveDocumentWithChunks(file.name, chunks);
      setActiveDocName(file.name);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `✅ Successfully indexed **${file.name}** (${chunks.length} chunks generated). You can now ask questions about this document!`,
        },
      ]);
    } catch (err) {
      console.error(err);
      alert(`Error processing PDF: ${err.message}`);
    } finally {
      setIsProcessingPdf(false);
    }
  };

  // Handle Asking Question with Streaming
  const handleSendMessage = async (e) => {
    e.preventDefault();
    const promptText = inputQuery.trim();
    if (!promptText || isGenerating) return;

    setInputQuery("");
    setIsGenerating(true);

    // 1. Append user prompt AND an empty assistant slot
    setMessages((prev) => [
      ...prev,
      { role: "user", content: promptText },
      { role: "assistant", content: "" },
    ]);

    try {
      // 2. Retrieve context from IndexedDB
      const retrieved = await retrieveRelevantChunks(promptText, 3);
      const contextText = formatContext(retrieved);

      let isFirstToken = true;

      // 3. Stream completion tokens
      await generateCompletionStream(promptText, contextText, (token) => {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const updated = [...prev];
          const lastIdx = updated.length - 1;

          if (isFirstToken) {
            updated[lastIdx] = { role: "assistant", content: token };
            isFirstToken = false;
          } else {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: (updated[lastIdx].content || "") + token,
            };
          }
          return updated;
        });
      });
    } catch (err) {
      console.error("Streaming error:", err);
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        updated[lastIdx] = {
          role: "assistant",
          content: `⚠️ **Error:** ${err.message || "An unexpected error occurred."}`,
        };
        return updated;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  // Clear indexed documents
  const handleClearDoc = async () => {
    try {
      await db.documents.clear();
      await db.chunks.clear();
    } catch (err) {
      console.error("Failed to clear DB:", err);
    }
    setActiveDocName(null);
    setMessages([]);
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-4">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-blue-400 flex items-center gap-2">
              📚 StudySync
            </h1>
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="p-2 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-white transition"
              title="BYOK Settings"
            >
              <Settings size={20} />
            </button>
          </div>

          {/* PDF Uploader Card */}
          <div className="space-y-3">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Document Workspace
            </label>

            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-2xl cursor-pointer bg-slate-950/50 hover:bg-slate-800/40 transition p-4 text-center">
              {isProcessingPdf ? (
                <div className="flex flex-col items-center text-blue-400 space-y-2">
                  <Loader2 className="animate-spin" size={24} />
                  <span className="text-xs font-medium">Indexing PDF...</span>
                </div>
              ) : (
                <div className="flex flex-col items-center text-slate-400 space-y-2">
                  <Upload size={24} />
                  <span className="text-xs font-medium text-slate-300">
                    Click to upload PDF notes
                  </span>
                  <span className="text-[10px] text-slate-500">
                    Processed 100% locally in browser
                  </span>
                </div>
              )}
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={handleFileUpload}
                disabled={isProcessingPdf}
              />
            </label>

            {/* Active Document Status */}
            {activeDocName && (
              <div className="flex items-center justify-between bg-slate-800/60 p-3 rounded-xl border border-slate-700 text-xs">
                <div className="flex items-center gap-2 truncate text-slate-200">
                  <FileText size={16} className="text-blue-400 shrink-0" />
                  <span className="truncate">{activeDocName}</span>
                </div>
                <button
                  onClick={handleClearDoc}
                  className="text-slate-400 hover:text-rose-400 transition"
                  title="Remove Document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Footer info */}
        <div className="text-[11px] text-slate-500 text-center border-t border-slate-800 pt-3">
          Privacy First • Zero-Server Architecture
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full bg-slate-950">
        {/* Chat Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-3">
              <div className="p-4 rounded-2xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <Bot size={32} />
              </div>
              <h2 className="text-xl font-bold text-slate-200">Welcome to StudySync Web</h2>
              <p className="text-sm text-slate-400">
                Upload a PDF note in the sidebar, set up your API key in Settings, and ask any question!
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-3 max-w-3xl ${
                  msg.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-sm font-bold ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-slate-800 text-blue-400 border border-slate-700"
                  }`}
                >
                  {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                </div>

                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed max-w-2xl ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-tr-none whitespace-pre-wrap"
                      : "bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-none shadow-lg"
                  }`}
                >
                  {msg.role === "user" ? (
                    msg.content || ""
                  ) : (
                    <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                      <ReactMarkdown>
                        {msg.content || "..."}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/50">
          <form
            onSubmit={handleSendMessage}
            className="max-w-3xl mx-auto flex items-center gap-2 bg-slate-950 border border-slate-800 focus-within:border-blue-500 rounded-2xl p-2 transition"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder={
                activeDocName
                  ? "Ask a question about your PDF..."
                  : "Upload a PDF first to ask context-aware questions..."
              }
              className="flex-1 bg-transparent px-3 text-sm text-slate-100 focus:outline-none"
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || isGenerating}
              className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 text-white transition"
            >
              <Send size={16} />
            </button>
          </form>
        </div>
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
}