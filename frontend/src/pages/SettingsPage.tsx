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

export function SettingsPage() {
  const { settings, saveSettings, clearSettings } = useSettings();
  const { healthCheck, error } = useApi();

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
  const [ollamaServerUrl, setOllamaServerUrl] = useState(
    settings.ollamaServerUrl || 'http://localhost:11434'
  );
  const [ollamaModel, setOllamaModel] = useState(
    settings.ollamaModel || 'llama3.2'
  );

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
    setOllamaServerUrl(
      settings.ollamaServerUrl || 'http://localhost:11434'
    );
    setOllamaModel(settings.ollamaModel || 'llama3.2');
  }, [settings]);

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

  const handleSave = () => {
    if (!endpointValidation.valid) {
      setSaveMessage('Please fix validation errors before saving');
      return;
    }

    saveSettings({
      apiEndpoint: apiEndpoint.trim(),
      apiKey: apiKey.trim() || undefined,
      defaultLanguage,
      defaultOutputFormat,
      defaultSampleRate: parseInt(defaultSampleRate),
      ollamaServerUrl: ollamaServerUrl.trim() || undefined,
      ollamaModel: ollamaModel.trim() || undefined,
    });

    setSaveMessage('Settings saved successfully!');
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
            Ollama Configuration
          </h2>

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
