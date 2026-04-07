/**
 * Main speech generation interface
 */

import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { useVoices } from '../hooks/useVoices';
import { useSettings } from '../hooks/useSettings';
import { SpeechSettings } from '../types/api';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Select, MultiSelect } from '../components/Select';
import { AudioPlayer } from '../components/AudioPlayer';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { SUPPORTED_LANGUAGES } from '../utils/languages';
import { formatVoiceLabel } from '../utils/format';
import { transcriptToGenerateFormat } from '../utils/transcript';

export function GeneratePage() {
  const [searchParams] = useSearchParams();
  const fromTranscriptId = searchParams.get('from');
  const { settings } = useSettings();
  const { voices, loading: voicesLoading } = useVoices();
  const { generateSpeechWithProgress, downloadAudio, getTranscript, loading, error } = useApi();

  const [transcript, setTranscript] = useState('');
  const [selectedSpeakers, setSelectedSpeakers] = useState<string[]>([]);
  const [requiredSpeakerCount, setRequiredSpeakerCount] = useState<number | null>(null);
  const [loadedTranscriptId, setLoadedTranscriptId] = useState<string | null>(null);
  const [speakerInstructions, setSpeakerInstructions] = useState<string[]>([]);
  const [speechSettings, setSpeechSettings] = useState<SpeechSettings>({
    language: settings.defaultLanguage,
    output_format: settings.defaultOutputFormat,
    sample_rate: settings.defaultSampleRate,
  });
  const [showSettings, setShowSettings] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFilename, setAudioFilename] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
    message: string;
  } | null>(null);

  // Initialize with example transcript (avoids Qwen3 TTS pitfalls: no greeting-style intros, no abbreviations)
  useEffect(() => {
    if (!fromTranscriptId && !transcript) {
      setTranscript(`Speaker 1: Run a quick check on the pipeline.
Speaker 2: Everything looks good from here.
Speaker 1: Output format is correct.
Speaker 2: Generation completed successfully.`);
    }
  }, [fromTranscriptId, transcript]);

  useEffect(() => {
    if (!fromTranscriptId) return;
    if (voicesLoading) return;
    if (loadedTranscriptId === fromTranscriptId) return;

    let cancelled = false;
    const hydrateFromTranscript = async () => {
      const item = await getTranscript(fromTranscriptId);
      if (!item || cancelled) return;

      const speakers = item.speakers || [];
      const speakerCount = speakers.length;
      const formattedTranscript = transcriptToGenerateFormat(item.transcript || [], speakers);

      setTranscript(formattedTranscript);
      setRequiredSpeakerCount(speakerCount > 0 ? speakerCount : null);

      if (speakerCount > 0) {
        const preselectedVoices = speakers.map((speaker) => {
          if (!speaker.voice_library_match) return '';
          const matched = voices.find((voice) => voice.id === speaker.voice_library_match);
          return matched?.name ?? '';
        });
        setSelectedSpeakers(preselectedVoices);
      } else {
        setSelectedSpeakers([]);
      }

      setLoadedTranscriptId(fromTranscriptId);
      setSuccessMessage(
        `Loaded transcript "${item.title}" with ${speakerCount} speaker${speakerCount === 1 ? '' : 's'}.`
      );
    };

    hydrateFromTranscript();
    return () => {
      cancelled = true;
    };
  }, [fromTranscriptId, getTranscript, loadedTranscriptId, voices, voicesLoading]);

  // Keep speaker instructions in sync with number of speakers
  useEffect(() => {
    setSpeakerInstructions((prev) => {
      const n = selectedSpeakers.length;
      if (prev.length === n) return prev;
      if (prev.length > n) return prev.slice(0, n);
      return [...prev, ...Array(n - prev.length).fill('')];
    });
  }, [selectedSpeakers.length]);

  const handleGenerate = async () => {
    if (!transcript.trim()) {
      setSuccessMessage(null);
      return;
    }

    if (requiredSpeakerCount !== null) {
      const hasMissingVoices =
        selectedSpeakers.length !== requiredSpeakerCount ||
        selectedSpeakers.some((speakerName) => !speakerName.trim());
      if (hasMissingVoices) {
        setSuccessMessage(null);
        return;
      }
    } else if (selectedSpeakers.length === 0) {
      setSuccessMessage(null);
      return;
    }

    setAudioUrl(null);
    setAudioFilename(null);
    setSuccessMessage(null);
    setProgress({ current: 0, total: 1, message: 'Starting...' });

    const request: Parameters<typeof generateSpeechWithProgress>[0] = {
      transcript,
      speakers: selectedSpeakers,
      settings: speechSettings,
    };
    if (speakerInstructions.some((s) => s.trim())) {
      request.speaker_instructions = speakerInstructions.map((s) => s.trim());
    }
    try {
      const response = await generateSpeechWithProgress(request, (cur: number, tot: number, msg: string) => {
        setProgress({ current: cur, total: tot || 1, message: msg });
      });

      if (response && response.audio_url) {
        const fullUrl = `${settings.apiEndpoint}${response.audio_url}`;
        setAudioUrl(fullUrl);
        setAudioFilename(response.audio_url.split('/').pop() || null);
        setSuccessMessage(response.message || 'Speech generated successfully!');
      }
    } finally {
      setProgress(null);
    }
  };

  const handleDownload = async () => {
    if (!audioFilename) return;

    setDownloading(true);
    const blob = await downloadAudio(audioFilename);
    setDownloading(false);

    if (blob) {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = audioFilename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
  };

  const speakerOptions = voices.map((voice) => ({
    value: voice.name,
    label: formatVoiceLabel(voice, { showQuality: true }),
  }));
  const requiredSpeakerOptions = [
    { value: '', label: 'Select a voice...' },
    ...speakerOptions,
  ];

  const languageOptions = SUPPORTED_LANGUAGES;

  const formatOptions = [
    { value: 'wav', label: 'WAV' },
  ];

  const sampleRateOptions = [
    { value: '24000', label: '24000 Hz' },
    { value: '44100', label: '44100 Hz' },
    { value: '48000', label: '48000 Hz' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Generate Speech</h1>
        <p className="mt-2 text-gray-600">Convert text to speech using AudioMesh</p>
      </div>

      {error && <Alert type="error" message={error} />}
      {successMessage && <Alert type="success" message={successMessage} />}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div>
          <Input
            label="Transcript"
            multiline
            rows={8}
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Enter transcript with speaker labels (e.g., 'Speaker 1: Hello')"
            required
          />
          <p className="mt-1 text-xs text-gray-500">
            Format: Use "Speaker 1:", "Speaker 2:", etc. to indicate different speakers
          </p>
        </div>

        <div>
          {voicesLoading ? (
            <div className="flex items-center gap-2">
              <LoadingSpinner size="sm" />
              <span className="text-sm text-gray-600">Loading voices...</span>
            </div>
          ) : requiredSpeakerCount !== null ? (
            <div className="space-y-3">
              <label className="block text-sm font-medium text-gray-700">
                Speakers
                <span className="text-red-500 ml-1">*</span>
              </label>
              <p className="text-xs text-gray-500">
                This transcript has {requiredSpeakerCount} speaker{requiredSpeakerCount === 1 ? '' : 's'}.
                Select one voice per speaker.
              </p>
              {Array.from({ length: requiredSpeakerCount }).map((_, index) => (
                <Select
                  key={`transcript-speaker-${index}`}
                  label={`Speaker ${index + 1} Voice`}
                  options={requiredSpeakerOptions}
                  value={selectedSpeakers[index] ?? ''}
                  onChange={(e) => {
                    const next = [...selectedSpeakers];
                    next[index] = e.target.value;
                    setSelectedSpeakers(next);
                  }}
                  error={
                    transcript.trim() && !(selectedSpeakers[index] || '').trim()
                      ? `Voice is required for Speaker ${index + 1}`
                      : undefined
                  }
                  required
                />
              ))}
            </div>
          ) : (
            <MultiSelect
              label="Speakers"
              options={speakerOptions}
              selected={selectedSpeakers}
              onChange={setSelectedSpeakers}
              required
              error={selectedSpeakers.length === 0 && transcript.trim() ? 'At least one speaker is required' : undefined}
            />
          )}
        </div>

        {selectedSpeakers.length > 0 && (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Style instructions (optional)</label>
            <p className="text-xs text-gray-500 mb-2">
              One per speaker. Use descriptive phrases for best results: &quot;Speak in an angry, frustrated tone&quot;,
              &quot;Sound excited and enthusiastic&quot;, or single words (angry, excited, calm, whisper).
            </p>
            {selectedSpeakers.map((name, i) => (
              <Input
                key={`${name || 'speaker'}-${i}`}
                label={
                  requiredSpeakerCount !== null
                    ? `Instruction for Speaker ${i + 1}${name ? ` (${name})` : ''}`
                    : `Instruction for ${name}`
                }
                value={speakerInstructions[i] ?? ''}
                onChange={(e) => {
                  const next = [...speakerInstructions];
                  next[i] = e.target.value;
                  setSpeakerInstructions(next);
                }}
                placeholder="e.g. angry, excited, or speak in a calm tone"
              />
            ))}
          </div>
        )}

        <div>
          <Button
            variant="secondary"
            onClick={() => setShowSettings(!showSettings)}
            className="mb-4"
          >
            {showSettings ? 'Hide' : 'Show'} Advanced Settings
          </Button>

          {showSettings && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
              <Select
                label="Language"
                options={languageOptions}
                value={speechSettings.language}
                onChange={(e) =>
                  setSpeechSettings({ ...speechSettings, language: e.target.value })
                }
              />
              <Select
                label="Output Format"
                options={formatOptions}
                value={speechSettings.output_format}
                onChange={(e) =>
                  setSpeechSettings({ ...speechSettings, output_format: e.target.value })
                }
              />
              <Select
                label="Sample Rate"
                options={sampleRateOptions}
                value={speechSettings.sample_rate.toString()}
                onChange={(e) =>
                  setSpeechSettings({
                    ...speechSettings,
                    sample_rate: parseInt(e.target.value),
                  })
                }
              />
            </div>
          )}
        </div>

        {progress && (
          <div className="space-y-2">
            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-600 transition-all duration-300"
                style={{
                  width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-sm text-gray-600">
              <span>{progress.message}</span>
              <span>
                {progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0}%
              </span>
            </div>
          </div>
        )}

        <Button
          variant="primary"
          onClick={handleGenerate}
          isLoading={loading}
          disabled={
            !transcript.trim() ||
            voicesLoading ||
            (requiredSpeakerCount !== null
              ? selectedSpeakers.length !== requiredSpeakerCount ||
                selectedSpeakers.some((speakerName) => !speakerName.trim())
              : selectedSpeakers.length === 0)
          }
          className="w-full"
        >
          Generate Speech
        </Button>
      </div>

      {audioUrl && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">Generated Audio</h2>
          <AudioPlayer src={audioUrl} filename={audioFilename || undefined} />
          <Button
            variant="secondary"
            onClick={handleDownload}
            isLoading={downloading}
            className="w-full"
          >
            Download Audio
          </Button>
        </div>
      )}
    </div>
  );
}