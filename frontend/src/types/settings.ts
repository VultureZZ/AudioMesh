/**
 * Settings interface for storing API endpoint, API key, and default preferences
 */

export type PrimaryLlmProvider = 'ollama' | 'openai';

export interface AppSettings {
  apiEndpoint: string;
  apiKey?: string;
  defaultLanguage: string;
  defaultOutputFormat: string;
  defaultSampleRate: number;
  /** Which LLM backs podcast script generation and segmentation (Ollama vs OpenAI Chat Completions). */
  primaryLlmProvider: PrimaryLlmProvider;
  /** OpenAI API key when primary provider is ChatGPT (stored locally; sent to your AudioMesh API only). */
  openaiApiKey?: string;
  /** Model id for OpenAI (e.g. gpt-4o-mini). */
  openaiModel?: string;
  ollamaServerUrl?: string;
  ollamaModel?: string;
  acestepConfigPath?: string;
  acestepLmModelPath?: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  apiEndpoint: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  apiKey: '',
  defaultLanguage: 'en',
  defaultOutputFormat: 'wav',
  defaultSampleRate: 24000,
  primaryLlmProvider: 'ollama',
  openaiApiKey: '',
  openaiModel: 'gpt-4o-mini',
  ollamaServerUrl: 'http://localhost:11434',
  ollamaModel: 'llama3.2',
  acestepConfigPath: 'acestep-v15-xl-sft',
  acestepLmModelPath: 'acestep-5Hz-lm-0.6B',
};