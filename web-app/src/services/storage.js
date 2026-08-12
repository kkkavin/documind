const DEFAULT_SETTINGS = {
  activeProvider: "gemini",
  providers: {
    gemini: {
      apiKey: "",
      model: "gemini-2.5-flash",
    },
    openai: {
      apiKey: "",
      model: "gpt-4o-mini",
    },
    anthropic: {
      apiKey: "",
      model: "claude-3-5-sonnet-20241022",
    },
    huggingface: {
      apiKey: "",
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    openrouter: {
      apiKey: "",
      model: "meta-llama/llama-3.3-70b-instruct",
    },
    local: {
      baseUrl: "http://localhost:11434/v1",
      model: "llama3.2",
    },
  },
};

const STORAGE_KEY = "documind_settings";

export const getSettings = () => {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (!saved) return DEFAULT_SETTINGS;
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
  } catch (e) {
    return DEFAULT_SETTINGS;
  }
};

export const saveSettings = (newSettings) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(newSettings));
};