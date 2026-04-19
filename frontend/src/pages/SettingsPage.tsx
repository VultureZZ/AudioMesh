/**
 * Settings configuration interface
 */

import { useState, useEffect } from 'react';
import { useSettings } from '../hooks/useSettings';
import { useApi } from '../hooks/useApi';
import { validateApiEndpoint } from '../utils/validation';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Select } from '../components/Select';
import { Alert } from '../components/Alert';
import { SUPPORTED_LANGUAGES } from '../utils/languages';
import type { PrimaryLlmProvider } from '../types/settings';

const PRIMARY_LLM_OPTIONS: Array<{ value: PrimaryLlmProvider; label: string }> = [
  { value: 'ollama', label: 'Ollama (local)' },
  { value: 'openai', label: 'ChatGPT (OpenAI API)' },
];

export function SettingsPage() {
  const { settings, saveSettings, clearSettings } = useSettings();
  const { healthCheck, error, getAceStepSettings, updateAceStepSettings, getAceStepModelCatalog } = useApi();

  const [apiEndpoint, setApiEndpoint] = useState(
    settings.apiEndpoint
  );
  const [apiKey, setApiKey] = useState(settings.apiKey || '');
  const [defaultLanguage, setDefaultLanguage] = useState(
    settings.defaultLanguage
  );
  const [defaultOutputFormat, setDefaultOutputFormat] = useState(
    settings.defaultOutputFormat
  );
  const [defaultSampleRate, setDefaultSampleRate] = useState(
    settings.defaultSampleRate.toString()
  );
  const [primaryLlmProvider, setPrimaryLlmProvider] = useState<PrimaryLlmProvider>(
    settings.primaryLlmProvider ?? 'ollama'
  );
  const [openaiApiKey, setOpenaiApiKey] = useState(settings.openaiApiKey || '');
  const [openaiModel, setOpenaiModel] = useState(
    settings.openaiModel || 'gpt-4o-mini'
  );
  const [ollamaServerUrl, setOllamaServerUrl] = useState(
    settings.ollamaServerUrl || 'http://localhost:11434'
  );
  const [ollamaModel, setOllamaModel] = useState(
    settings.ollamaModel || 'llama3.2'
  );
  const [acestepConfigPath, setAcestepConfigPath] = useState(
    settings.acestepConfigPath || 'acestep-v15-turbo'
  );
  const [acestepLmModelPath, setAcestepLmModelPath] = useState(
    settings.acestepLmModelPath || 'acestep-5Hz-lm-0.6B'
  );
  const [acestepDitModelOptions, setAcestepDitModelOptions] = useState<
    Array<{ value: string; label: string }>
  >([{ value: 'acestep-v15-turbo', label: 'acestep-v15-turbo' }]);
  const [acestepLmModelOptions, setAcestepLmModelOptions] = useState<
    Array<{ value: string; label: string }>
  >([{ value: 'acestep-5Hz-lm-0.6B', label: 'acestep-5Hz-lm-0.6B' }]);

  const [testStatus, setTestStatus] = useState<
    'idle' | 'testing' | 'success' | 'error'
  >('idle');
  const [testMessage, setTestMessage] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    setApiEndpoint(settings.apiEndpoint);
    setApiKey(settings.apiKey || '');
    setDefaultLanguage(settings.defaultLanguage);
    setDefaultOutputFormat(settings.defaultOutputFormat);
    setDefaultSampleRate(settings.defaultSampleRate.toString());
    setPrimaryLlmProvider(settings.primaryLlmProvider ?? 'ollama');
    setOpenaiApiKey(settings.openaiApiKey || '');
    setOpenaiModel(settings.openaiModel || 'gpt-4o-mini');
    setOllamaServerUrl(
      settings.ollamaServerUrl || 'http://localhost:11434'
    );
    setOllamaModel(settings.ollamaModel || 'llama3.2');
    setAcestepConfigPath(settings.acestepConfigPath || 'acestep-v15-turbo');
    setAcestepLmModelPath(settings.acestepLmModelPath || 'acestep-5Hz-lm-0.6B');
  }, [settings]);

  useEffect(() => {
    const loadAceStepSettings = async () => {
      const catalog = await getAceStepModelCatalog();
      if (catalog) {
        if (catalog.dit_models?.length > 0) {
          setAcestepDitModelOptions(
            catalog.dit_models.map((model) => ({ value: model, label: model }))
          );
        }
        if (catalog.lm_models?.length > 0) {
          setAcestepLmModelOptions(
            catalog.lm_models.map((model) => ({ value: model, label: model }))
          );
        }
      }

      const currentSettings = await getAceStepSettings();
      if (currentSettings) {
        setAcestepConfigPath(currentSettings.acestep_config_path);
        setAcestepLmModelPath(currentSettings.acestep_lm_model_path);
      } else if (catalog?.current) {
        const dit = catalog.current.acestep_config_path;
        const lm = catalog.current.acestep_lm_model_path;
        if (dit) setAcestepConfigPath(dit);
        if (lm) setAcestepLmModelPath(lm);
      }
    };
    loadAceStepSettings();
  }, [getAceStepModelCatalog, getAceStepSettings]);

  const endpointValidation = validateApiEndpoint(apiEndpoint);

  const handleTestConnection = async () => {
    setTestStatus('testing');
    setTestMessage(null);

    // Temporarily update API client config
    const originalEndpoint = settings.apiEndpoint;
    const originalKey = settings.apiKey;

    saveSettings({
      ...settings,
      apiEndpoint,
      apiKey: apiKey || undefined,
    });

    const response = await healthCheck();

    if (response) {
      setTestStatus('success');
      setTestMessage(
        `Connected successfully! Service: ${response.service}, Version: ${response.version}`
      );
    } else {
      setTestStatus('error');
      setTestMessage(error || 'Connection failed');
    }

    // Restore original settings
    saveSettings({
      ...settings,
      apiEndpoint: originalEndpoint,
      apiKey: originalKey,
    });
  };

  const handleSave = async () => {
    if (!endpointValidation.valid) {
      setSaveMessage('Please fix validation errors before saving');
      return;
    }

    const runtimeSettings = await updateAceStepSettings({
      acestep_config_path: acestepConfigPath,
      acestep_lm_model_path: acestepLmModelPath,
    });
    if (!runtimeSettings) {
      setSaveMessage('Failed to save ACE-Step settings');
      return;
    }

    saveSettings({
      apiEndpoint: apiEndpoint.trim(),
      apiKey: apiKey.trim() || undefined,
      defaultLanguage,
      defaultOutputFormat,
      defaultSampleRate: parseInt(defaultSampleRate),
      primaryLlmProvider,
      openaiApiKey: openaiApiKey.trim() || undefined,
      openaiModel: openaiModel.trim() || undefined,
      ollamaServerUrl: ollamaServerUrl.trim() || undefined,
      ollamaModel: ollamaModel.trim() || undefined,
      acestepConfigPath: acestepConfigPath.trim(),
      acestepLmModelPath: acestepLmModelPath.trim(),
    });

    setSaveMessage(
      runtimeSettings.restart_required
        ? 'Settings saved. ACE-Step will restart on the next music request.'
        : 'Settings saved successfully!'
    );
    setTimeout(() => setSaveMessage(null), 3000);
  };

  const handleClear = () => {
    if (
      confirm(
        'Are you sure you want to clear all settings? This will reset to defaults.'
      )
    ) {
      clearSettings();
      setSaveMessage('Settings cleared!');
      setTimeout(() => setSaveMessage(null), 3000);
    }
  };

  const languageOptions = SUPPORTED_LANGUAGES;

  const formatOptions = [{ value: 'wav', label: 'WAV' }];

  const sampleRateOptions = [
    { value: '24000', label: '24000 Hz' },
    { value: '44100', label: '44100 Hz' },
    { value: '48000', label: '48000 Hz' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="mt-2 text-gray-600">
          Configure API connection and default preferences
        </p>
      </div>

      {saveMessage && (
        <Alert
          type={saveMessage.includes('success') ? 'success' : 'info'}
          message={saveMessage}
          onClose={() => setSaveMessage(null)}
        />
      )}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            API Configuration
          </h2>

          <div className="space-y-4">
            <Input
              label="API Endpoint"
              type="url"
              value={apiEndpoint}
              onChange={(e) => setApiEndpoint(e.target.value)}
              error={endpointValidation.error}
              placeholder="http://localhost:8000"
              required
            />

            <Input
              label="API Key (Optional)"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Leave empty if no API key is required"
            />

            <div className="flex gap-2">
              <Button
                variant="secondary"
                onClick={handleTestConnection}
                isLoading={testStatus === 'testing'}
              >
                Test Connection
              </Button>

              {testStatus === 'success' && testMessage && (
                <Alert type="success" message={testMessage} />
              )}
              {testStatus === 'error' && testMessage && (
                <Alert type="error" message={testMessage} />
              )}
            </div>
          </div>
        </div>

        <div className="border-t pt-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Text generation (LLM)
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Choose whether podcast script generation and script segmentation use your local Ollama server
            or the OpenAI API. Production mode still uses Ollama for the Director step; keep the Ollama
            server below configured and running when using production features.
          </p>
          <div className="space-y-4">
            <Select
              label="Primary provider"
              options={PRIMARY_LLM_OPTIONS}
              value={primaryLlmProvider}
              onChange={(e) =>
                setPrimaryLlmProvider(e.target.value as PrimaryLlmProvider)
              }
            />
            {primaryLlmProvider === 'openai' && (
              <>
                <Input
                  label="OpenAI API key"
                  type="password"
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                  placeholder="sk-..."
                />
                <Input
                  label="OpenAI model"
                  type="text"
                  value={openaiModel}
                  onChange={(e) => setOpenaiModel(e.target.value)}
                  placeholder="gpt-4o-mini"
                />
              </>
            )}
          </div>
        </div>

        <div className="border-t pt-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Ollama Configuration
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Used when Primary provider is Ollama, and for Production Director / prosody when production
            mode is enabled.
          </p>

          <div className="space-y-4">
            <Input
              label="Ollama Server URL"
              type="url"
              value={ollamaServerUrl}
              onChange={(e) => setOllamaServerUrl(e.target.value)}
              placeholder="http://localhost:11434"
            />

            <Input
              label="Ollama Model"
              type="text"
              value={ollamaModel}
              onChange={(e) => setOllamaModel(e.target.value)}
              placeholder="llama3.2"
            />
          </div>
        </div>

        <div className="border-t pt-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            ACE-Step Music Configuration
          </h2>

          <div className="space-y-4">
            <Select
              label="ACE-Step DiT Model"
              options={acestepDitModelOptions}
              value={acestepConfigPath}
              onChange={(e) => setAcestepConfigPath(e.target.value)}
            />

            <Select
              label="ACE-Step LM Model"
              options={acestepLmModelOptions}
              value={acestepLmModelPath}
              onChange={(e) => setAcestepLmModelPath(e.target.value)}
            />
            <p className="text-sm text-gray-600">
              Missing models are auto-downloaded by ACE-Step on first use.
            </p>
          </div>
        </div>

        <div className="border-t pt-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Default Settings
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Select
              label="Default Language"
              options={languageOptions}
              value={defaultLanguage}
              onChange={(e) => setDefaultLanguage(e.target.value)}
            />

            <Select
              label="Default Output Format"
              options={formatOptions}
              value={defaultOutputFormat}
              onChange={(e) => setDefaultOutputFormat(e.target.value)}
            />

            <Select
              label="Default Sample Rate"
              options={sampleRateOptions}
              value={defaultSampleRate}
              onChange={(e) => setDefaultSampleRate(e.target.value)}
            />
          </div>
        </div>

        <div className="border-t pt-6 flex gap-4">
          <Button
            variant="primary"
            onClick={handleSave}
            disabled={!endpointValidation.valid}
          >
            Save Settings
          </Button>

          <Button variant="secondary" onClick={handleClear}>
            Clear Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
