/**
 * Settings interface for storing API endpoint, API key, and default preferences
 */

export interface AppSettings {
  apiEndpoint: string;
  apiKey?: string;
  defaultLanguage: string;
  defaultOutputFormat: string;
  defaultSampleRate: number;
  ollamaServerUrl?: string;
  ollamaModel?: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  apiEndpoint: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  apiKey: '',
  defaultLanguage: 'en',
  defaultOutputFormat: 'wav',
  defaultSampleRate: 24000,
  ollamaServerUrl: 'http://localhost:11434',
  ollamaModel: 'llama3.2',
};