/**
 * Article Podcaster page - Convert articles to podcasts
 */

import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { useVoices } from '../hooks/useVoices';
import { useSettings } from '../hooks/useSettings';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Select, MultiSelect } from '../components/Select';
import { AudioPlayer } from '../components/AudioPlayer';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { formatVoiceLabel } from '../utils/format';

const GENRES = [
  { value: 'Comedy', label: 'Comedy' },
  { value: 'Serious', label: 'Serious' },
  { value: 'News', label: 'News' },
  { value: 'Educational', label: 'Educational' },
  { value: 'Storytelling', label: 'Storytelling' },
  { value: 'Interview', label: 'Interview' },
  { value: 'Documentary', label: 'Documentary' },
];

const DURATIONS = [
  { value: '5 min', label: '5 minutes' },
  { value: '10 min', label: '10 minutes' },
  { value: '15 min', label: '15 minutes' },
  { value: '30 min', label: '30 minutes' },
];

export function PodcastPage() {
  const { settings } = useSettings();
  const { voices, loading: voicesLoading } = useVoices();
  const {
    generatePodcastScript,
    generatePodcastAudio,
    downloadPodcastAudio,
    downloadPodcastById,
    loading,
    error,
  } = useApi();

  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [selectedVoices, setSelectedVoices] = useState<string[]>([]);
  const [genre, setGenre] = useState('News');
  const [duration, setDuration] = useState('10 min');
  const [saveToLibrary, setSaveToLibrary] = useState(true);
  const [script, setScript] = useState<string | null>(null);
  const [isEditingScript, setIsEditingScript] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFilename, setAudioFilename] = useState<string | null>(null);
  const [podcastId, setPodcastId] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [downloading, setDownloading] = useState(false);

  const speakerOptions = voices.map((voice) => ({
    value: voice.name,
    label: formatVoiceLabel(voice, { showQuality: true }),
  }));

  const handleGenerateScript = async () => {
    if (!url.trim()) {
      setSuccessMessage(null);
      return;
    }

    if (selectedVoices.length === 0) {
      setSuccessMessage(null);
      return;
    }

    if (selectedVoices.length > 4) {
      setSuccessMessage(null);
      return;
    }

    setScript(null);
    setAudioUrl(null);
    setAudioFilename(null);
    setPodcastId(null);
    setSuccessMessage(null);
    setWarnings([]);
    setIsEditingScript(false);

    const response = await generatePodcastScript({
      url: url.trim(),
      voices: selectedVoices,
      genre,
      duration,
      ollama_url: settings.ollamaServerUrl,
      ollama_model: settings.ollamaModel,
    });

    if (response && response.script) {
      setScript(response.script);
      setSuccessMessage('Script generated successfully! You can review and edit it before generating audio.');
      setWarnings(response.warnings || []);
    }
  };

  const handleEditScript = () => {
    setIsEditingScript(true);
  };

  const handleGenerateAudio = async () => {
    if (!script || !script.trim()) {
      setSuccessMessage(null);
      return;
    }

    if (selectedVoices.length === 0) {
      setSuccessMessage(null);
      return;
    }

    setAudioUrl(null);
    setAudioFilename(null);
    setPodcastId(null);
    setSuccessMessage(null);
    setWarnings([]);

    const response = await generatePodcastAudio({
      script: script.trim(),
      voices: selectedVoices,
      title: title.trim() || undefined,
      source_url: url.trim() || undefined,
      genre,
      duration,
      save_to_library: saveToLibrary,
    });

    if (response && response.audio_url) {
      const fullUrl = `${settings.apiEndpoint}${response.audio_url}`;
      setAudioUrl(fullUrl);
      setPodcastId(response.podcast_id || null);
      // Only set a filename when we are using the legacy filename-download endpoint
      if (response.audio_url.includes('/api/v1/podcast/download/')) {
        setAudioFilename(response.audio_url.split('/').pop() || null);
      } else {
        setAudioFilename(null);
      }
      setSuccessMessage(
        response.podcast_id
          ? 'Podcast audio generated and saved to the library!'
          : 'Podcast audio generated successfully!'
      );
      setWarnings(response.warnings || []);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    let blob: Blob | null = null;

    if (podcastId) {
      blob = await downloadPodcastById(podcastId);
    } else if (audioFilename) {
      blob = await downloadPodcastAudio(audioFilename);
    }

    setDownloading(false);

    if (blob) {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const fileBase = title.trim() || podcastId || audioFilename || 'podcast';
      a.download = `${fileBase}.wav`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Article Podcaster</h1>
        <p className="mt-2 text-gray-600">Convert articles into multi-voice podcasts using AI</p>
      </div>

      {error && <Alert type="error" message={error} />}
      {successMessage && <Alert type="success" message={successMessage} />}
      {warnings.length > 0 && (
        <Alert
          type="warning"
          message={`Warnings: ${warnings.join(' • ')}`}
          onClose={() => setWarnings([])}
        />
      )}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div>
          <Input
            label="Title (Optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., Weekly Tech Briefing"
          />
          <p className="mt-1 text-xs text-gray-500">Used when saving to the Podcast Library</p>
        </div>

        <div>
          <Input
            label="Article URL"
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/article"
            required
          />
          <p className="mt-1 text-xs text-gray-500">
            Enter the URL of the article you want to convert to a podcast
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            {voicesLoading ? (
              <div className="flex items-center gap-2">
                <LoadingSpinner size="sm" />
                <span className="text-sm text-gray-600">Loading voices...</span>
              </div>
            ) : (
              <MultiSelect
                label="Voices (up to 4)"
                options={speakerOptions}
                selected={selectedVoices}
                onChange={(voices) => {
                  if (voices.length <= 4) {
                    setSelectedVoices(voices);
                  }
                }}
                required
                error={
                  selectedVoices.length === 0
                    ? 'At least one voice is required'
                    : selectedVoices.length > 4
                      ? 'Maximum 4 voices allowed'
                      : undefined
                }
              />
            )}
          </div>

          <div>
            <Select
              label="Genre"
              options={GENRES}
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              required
            />
          </div>

          <div>
            <Select
              label="Duration"
              options={DURATIONS}
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              required
            />
          </div>
        </div>

        <div className="flex items-center justify-between bg-gray-50 border rounded-lg p-4">
          <div>
            <p className="text-sm font-medium text-gray-900">Save to Podcast Library</p>
            <p className="text-xs text-gray-500">If enabled, the generated audio will be saved and appear in Podcast Library</p>
          </div>
          <label className="inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only"
              checked={saveToLibrary}
              onChange={(e) => setSaveToLibrary(e.target.checked)}
            />
            <div
              className={`w-11 h-6 rounded-full transition-colors ${
                saveToLibrary ? 'bg-primary-600' : 'bg-gray-300'
              }`}
            >
              <div
                className={`w-5 h-5 bg-white rounded-full shadow transform transition-transform translate-y-0.5 ${
                  saveToLibrary ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </div>
          </label>
        </div>

        <div className="flex gap-4">
          <Button
            variant="primary"
            onClick={handleGenerateScript}
            isLoading={loading}
            disabled={
              !url.trim() ||
              selectedVoices.length === 0 ||
              selectedVoices.length > 4 ||
              voicesLoading
            }
            className="flex-1"
          >
            Generate Script
          </Button>
        </div>
      </div>

      {script && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold text-gray-900">Generated Script</h2>
            <div className="flex gap-2">
              {!isEditingScript && (
                <Button variant="secondary" onClick={handleEditScript}>
                  Edit Script
                </Button>
              )}
              {isEditingScript && (
                <Button variant="secondary" onClick={() => setIsEditingScript(false)}>
                  Done Editing
                </Button>
              )}
            </div>
          </div>

          <Input
            label=""
            multiline
            rows={12}
            value={script}
            onChange={(e) => setScript(e.target.value)}
            disabled={!isEditingScript}
            className={isEditingScript ? '' : 'bg-gray-50'}
          />

          <Button
            variant="primary"
            onClick={handleGenerateAudio}
            isLoading={loading}
            disabled={!script.trim() || selectedVoices.length === 0}
            className="w-full"
          >
            Generate Audio
          </Button>
        </div>
      )}

      {audioUrl && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">Generated Podcast</h2>
          <AudioPlayer src={audioUrl} filename={title.trim() || audioFilename || undefined} />
          <Button
            variant="secondary"
            onClick={handleDownload}
            isLoading={downloading}
            disabled={!podcastId && !audioFilename}
            className="w-full"
          >
            Download Audio
          </Button>
        </div>
      )}
    </div>
  );
}
