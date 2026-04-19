/**
 * Article Podcaster page - Convert articles to podcasts
 */

import { useEffect, useState } from 'react';
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

const PRODUCTION_STYLES = [
  { value: 'tech_talk', label: 'Tech Talk' },
  { value: 'casual', label: 'Casual Chat' },
  { value: 'news', label: 'News' },
  { value: 'storytelling', label: 'Storytelling' },
];

export function PodcastPage() {
  const { settings } = useSettings();
  const { voices, loading: voicesLoading } = useVoices();
  const {
    generatePodcastScript,
    generatePodcastAudio,
    generatePodcastProduction,
    getPodcastProductionStatus,
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
  const [productionMode, setProductionMode] = useState(false);
  const [productionStyle, setProductionStyle] = useState<
    'tech_talk' | 'casual' | 'news' | 'storytelling'
  >('casual');
  const [cueIntro, setCueIntro] = useState(true);
  const [cueBed, setCueBed] = useState(false);
  const [cueTransitions, setCueTransitions] = useState(true);
  const [cueOutro, setCueOutro] = useState(true);
  const [script, setScript] = useState<string | null>(null);
  const [isEditingScript, setIsEditingScript] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFilename, setAudioFilename] = useState<string | null>(null);
  const [podcastId, setPodcastId] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [downloading, setDownloading] = useState(false);
  const [productionTaskId, setProductionTaskId] = useState<string | null>(null);
  const [productionTaskStatus, setProductionTaskStatus] = useState<string | null>(null);
  const [productionStageProgress, setProductionStageProgress] = useState<Record<string, string>>({});
  const [productionCueStatus, setProductionCueStatus] = useState<Record<string, string>>({});
  const isProductionTaskActive =
    !!productionTaskId && !!productionTaskStatus && ['queued', 'running'].includes(productionTaskStatus);

  const speakerOptions = voices.map((voice) => ({
    value: voice.name,
    label: formatVoiceLabel(voice, { showQuality: true }),
  }));

  useEffect(() => {
    if (!productionTaskId || !productionTaskStatus) return;
    if (!['queued', 'running'].includes(productionTaskStatus)) return;

    const interval = window.setInterval(async () => {
      const status = await getPodcastProductionStatus(productionTaskId);
      if (!status) return;
      setProductionTaskStatus(status.status);
      setProductionStageProgress(status.stage_progress || {});
      setProductionCueStatus(status.cue_status || {});
      setWarnings(status.warnings || []);

      if (status.status === 'succeeded' && status.audio_url) {
        const fullUrl = `${settings.apiEndpoint}${status.audio_url}`;
        setAudioUrl(fullUrl);
        setPodcastId(status.podcast_id || null);
        if (status.audio_url.includes('/api/v1/podcast/download/')) {
          setAudioFilename(status.audio_url.split('/').pop() || null);
        } else {
          setAudioFilename(null);
        }
        setSuccessMessage(
          status.podcast_id
            ? 'Production podcast generated and saved to the library!'
            : 'Production podcast generated successfully!'
        );
      }
    }, 2500);

    return () => window.clearInterval(interval);
  }, [productionTaskId, productionTaskStatus, getPodcastProductionStatus, settings.apiEndpoint]);

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
    setProductionTaskId(null);
    setProductionTaskStatus(null);
    setProductionStageProgress({});
    setProductionCueStatus({});

    if (productionMode) {
      const enabledCues: ('intro' | 'outro' | 'transitions' | 'bed')[] = [];
      if (cueIntro) enabledCues.push('intro');
      if (cueBed) enabledCues.push('bed');
      if (cueTransitions) enabledCues.push('transitions');
      if (cueOutro) enabledCues.push('outro');

      const submit = await generatePodcastProduction({
        script: script.trim(),
        voices: selectedVoices,
        title: title.trim() || undefined,
        source_url: url.trim() || undefined,
        genre,
        duration,
        save_to_library: saveToLibrary,
        production_mode: true,
        style: productionStyle,
        enabled_cues: enabledCues,
        ollama_url: settings.ollamaServerUrl,
        ollama_model: settings.ollamaModel,
      });
      if (submit?.task_id) {
        setProductionTaskId(submit.task_id);
        setProductionTaskStatus(submit.status);
        setSuccessMessage('Production task submitted. Processing...');
      }
      return;
    }

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
      const inferredExt = audioFilename?.split('.').pop();
      const ext = inferredExt ? `.${inferredExt}` : productionMode ? '.mp3' : '.wav';
      a.download = `${fileBase}${ext}`;
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
            <p className="text-xs text-gray-500">
              If enabled, the generated audio will be saved and appear in Podcast Library
            </p>
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

        <div className="flex items-center justify-between bg-gray-50 border rounded-lg p-4">
          <div>
            <p className="text-sm font-medium text-gray-900">Production Mode</p>
            <p className="text-xs text-gray-500">
              Generate intro/outro/transition/bed music and mix with the voice track
            </p>
          </div>
          <label className="inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only"
              checked={productionMode}
              onChange={(e) => setProductionMode(e.target.checked)}
            />
            <div
              className={`w-11 h-6 rounded-full transition-colors ${
                productionMode ? 'bg-primary-600' : 'bg-gray-300'
              }`}
            >
              <div
                className={`w-5 h-5 bg-white rounded-full shadow transform transition-transform translate-y-0.5 ${
                  productionMode ? 'translate-x-5' : 'translate-x-1'
                }`}
              />
            </div>
          </label>
        </div>

        {productionMode && (
          <div className="bg-gray-50 border rounded-lg p-4 space-y-3">
            <Select
              label="Style"
              options={PRODUCTION_STYLES}
              value={productionStyle}
              onChange={(e) =>
                setProductionStyle(e.target.value as 'tech_talk' | 'casual' | 'news' | 'storytelling')
              }
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={cueIntro} onChange={(e) => setCueIntro(e.target.checked)} />
                Intro Music
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={cueBed} onChange={(e) => setCueBed(e.target.checked)} />
                Background Bed
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={cueTransitions}
                  onChange={(e) => setCueTransitions(e.target.checked)}
                />
                Transition Stings
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={cueOutro} onChange={(e) => setCueOutro(e.target.checked)} />
                Outro Music
              </label>
            </div>
          </div>
        )}

        <div className="flex gap-4">
          <Button
            variant="primary"
            onClick={handleGenerateScript}
            isLoading={loading}
            disabled={
              loading ||
              isProductionTaskActive ||
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
            disabled={loading || isProductionTaskActive || !script.trim() || selectedVoices.length === 0}
            className="w-full"
          >
            {productionMode ? 'Generate Production Audio' : 'Generate Audio'}
          </Button>
        </div>
      )}

      {productionTaskId && (
        <div className="bg-white rounded-lg shadow p-6 space-y-3">
          <h2 className="text-xl font-semibold text-gray-900">Production Progress</h2>
          <p className="text-sm text-gray-600">
            Task <span className="font-mono">{productionTaskId}</span> ({productionTaskStatus || 'queued'})
          </p>
          <div className="text-sm text-gray-700 space-y-1">
            <p>Stage 1: Generating Script — {productionStageProgress.generating_script || 'pending'}</p>
            <p>
              Stage 2: Generating Voice Track — {productionStageProgress.generating_voice_track || 'pending'}
            </p>
            <p>
              Stage 3: Generating production stems (music / SFX) —{' '}
              {productionStageProgress.generating_music_cues || 'pending'}
            </p>
            {Object.keys(productionCueStatus).length > 0 && (
              <div className="pl-2 text-xs text-gray-600 space-y-1">
                {Object.entries(productionCueStatus).map(([cue, state]) => (
                  <p key={cue}>
                    {cue}: {state}
                  </p>
                ))}
              </div>
            )}
            <p>
              Stage 4: Mixing Production Audio — {productionStageProgress.mixing_production_audio || 'pending'}
            </p>
            <p>Stage 5: Ready to Download — {productionStageProgress.ready_to_download || 'pending'}</p>
          </div>
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
