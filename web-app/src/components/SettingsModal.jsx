import { useState } from "react";
import { X, Check, Key, Server, Cpu } from "lucide-react";
import { getSettings, saveSettings } from "../services/storage";

const PROVIDERS = [
  { id: "gemini", name: "Google Gemini", icon: Cpu },
  { id: "openai", name: "OpenAI", icon: Key },
  { id: "anthropic", name: "Anthropic", icon: Key },
  { id: "huggingface", name: "Hugging Face", icon: Server },
  { id: "openrouter", name: "OpenRouter", icon: Server },
  { id: "local", name: "Local Ollama", icon: Server },
];

export default function SettingsModal({ isOpen, onClose }) {
  const [settings, setSettings] = useState(getSettings());
  const [activeTab, setActiveTab] = useState(settings.activeProvider);
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const handleProviderChange = (field, value) => {
    setSettings((prev) => ({
      ...prev,
      providers: {
        ...prev.providers,
        [activeTab]: {
          ...prev.providers[activeTab],
          [field]: value,
        },
      },
    }));
  };

  const handleSave = () => {
    const updated = { ...settings, activeProvider: activeTab };
    saveSettings(updated);
    setSettings(updated);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  const currentConfig = settings.providers[activeTab];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-2xl rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between gap-3 px-4 md:px-6 py-3 md:py-4 border-b border-slate-800">
          <h2 className="text-lg md:text-xl font-semibold text-white flex items-center gap-2">
            ⚙️ Model Provider & BYOK Settings
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition shrink-0">
            <X size={20} />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden flex-col md:flex-row">
          {/* Sidebar Tabs */}
          <div className="grid grid-cols-2 gap-1.5 md:flex md:flex-col md:gap-1 w-full md:w-48 bg-slate-950/50 p-3 border-b md:border-b-0 border-r-0 md:border-r border-slate-800">
            {PROVIDERS.map((p) => {
              const Icon = p.icon;
              const isActive = activeTab === p.p_id || activeTab === p.id;
              const isSelectedProvider = settings.activeProvider === p.id;

              return (
                <button
                  key={p.id}
                  onClick={() => setActiveTab(p.id)}
                  className={`w-full flex items-center justify-between px-2 py-1.5 md:px-3 md:py-2.5 rounded-xl text-xs md:text-sm font-medium transition ${
                    isActive
                      ? "bg-blue-600 text-white"
                      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                  }`}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Icon size={16} className="shrink-0" />
                    <span className="truncate">{p.name}</span>
                  </span>
                  {isSelectedProvider && (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" title="Active Model" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Form Content */}
          <div className="flex-1 min-w-0 p-2.5 md:p-6 space-y-2 md:space-y-5 overflow-y-auto">
            <div className="flex items-center justify-between gap-2 flex-wrap bg-slate-800/40 p-2 md:p-3 rounded-xl border border-slate-800">
              <span className="text-sm font-medium text-slate-300">Set as Active Provider</span>
              <button
                onClick={() => {
                  setSettings((prev) => ({ ...prev, activeProvider: activeTab }));
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  settings.activeProvider === activeTab
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                }`}
              >
                {settings.activeProvider === activeTab ? "Active" : "Use This Model"}
              </button>
            </div>

            {/* API Key Field (if not local) */}
            {activeTab !== "local" ? (
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  API Key ({activeTab})
                </label>
                <input
                  type="password"
                  value={currentConfig.apiKey || ""}
                  onChange={(e) => handleProviderChange("apiKey", e.target.value)}
                  placeholder={`Enter your ${activeTab} API key...`}
                  className="w-full px-3 py-1.5 md:px-4 md:py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            ) : (
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Ollama Base URL
                </label>
                <input
                  type="text"
                  value={currentConfig.baseUrl || "http://localhost:11434/v1"}
                  onChange={(e) => handleProviderChange("baseUrl", e.target.value)}
                  placeholder="http://localhost:11434/v1"
                  className="w-full px-3 py-1.5 md:px-4 md:py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            )}

            {/* Model Name Field */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Model Name
              </label>
              <input
                type="text"
                value={currentConfig.model || ""}
                onChange={(e) => handleProviderChange("model", e.target.value)}
                placeholder="Model ID (e.g. meta-llama/Meta-Llama-3-8B-Instruct)"
                className="w-full px-3 py-1.5 md:px-4 md:py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 md:px-6 py-3 md:py-4 border-t border-slate-800 bg-slate-950/40">
          <span className="text-xs text-slate-500">
            Keys are saved locally in your browser context.
          </span>
          <div className="flex items-center gap-3">
            {savedSuccess && (
              <span className="text-xs text-emerald-400 flex items-center gap-1 font-medium">
                <Check size={14} /> Saved!
              </span>
            )}
            <button
              onClick={handleSave}
              className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition"
            >
              Save Configuration
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}