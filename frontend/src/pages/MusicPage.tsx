/**
 * Music generation page using ACE-Step backend integration.
 */

import { useEffect, useMemo, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { useSettings } from '../hooks/useSettings';
import { MusicHistoryItem, MusicPreset } from '../types/api';
import { AudioPlayer } from '../components/AudioPlayer';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Select } from '../components/Select';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';

type MusicTab = 'simple' | 'custom';
type SimpleInputMode = 'refine' | 'exact';

const GENRES = [
  { value: 'Pop', label: 'Pop' },
  { value: 'Rock', label: 'Rock' },
  { value: 'Hip Hop', label: 'Hip Hop' },
  { value: 'Electronic', label: 'Electronic' },
  { value: 'R&B', label: 'R&B' },
  { value: 'Jazz', label: 'Jazz' },
  { value: 'Cinematic', label: 'Cinematic' },
];

const MOODS = [
  { value: 'Neutral', label: 'Neutral' },
  { value: 'Happy', label: 'Happy' },
  { value: 'Melancholic', label: 'Melancholic' },
  { value: 'Energetic', label: 'Energetic' },
  { value: 'Dark', label: 'Dark' },
  { value: 'Dreamy', label: 'Dreamy' },
];

const LANGUAGE_OPTIONS = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'zh', label: 'Chinese' },
];

const KEY_OPTIONS = [
  { value: '', label: 'Auto' },
  { value: 'C Major', label: 'C Major' },
  { value: 'G Major', label: 'G Major' },
  { value: 'D Major', label: 'D Major' },
  { value: 'A Major', label: 'A Major' },
  { value: 'E minor', label: 'E minor' },
  { value: 'A minor', label: 'A minor' },
  { value: 'D minor', label: 'D minor' },
];

const TIME_SIGNATURE_OPTIONS = [
  { value: '', label: 'Auto' },
  { value: '4', label: '4/4' },
  { value: '3', label: '3/4' },
  { value: '2', label: '2/4' },
  { value: '6', label: '6/8' },
];

const AUDIO_FORMAT_OPTIONS = [
  { value: 'mp3', label: 'MP3' },
  { value: 'wav', label: 'WAV' },
  { value: 'flac', label: 'FLAC' },
];

export function MusicPage() {
  const { settings } = useSettings();
  const {
    generateMusic,
    generateMusicCover,
    simpleGenerateMusic,
    getMusicStatus,
    generateLyrics,
    downloadMusic,
    checkMusicHealth,
    listMusicPresets,
    createMusicPreset,
    updateMusicPreset,
    deleteMusicPreset,
    listMusicHistory,
    deleteMusicHistoryItem,
    loading,
    error,
  } = useApi();

  const [activeTab, setActiveTab] = useState<MusicTab>('simple');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [simpleDescription, setSimpleDescription] = useState('');
  const [simpleInputMode, setSimpleInputMode] = useState<SimpleInputMode>('refine');
  const [simpleInstrumental, setSimpleInstrumental] = useState(false);
  const [simpleLanguage, setSimpleLanguage] = useState('en');
  const [simpleDuration, setSimpleDuration] = useState('60');
  const [simpleBatchSize, setSimpleBatchSize] = useState('1');
  const [simpleExactCaption, setSimpleExactCaption] = useState('');
  const [simpleExactLyrics, setSimpleExactLyrics] = useState('');
  const [simpleExactBpm, setSimpleExactBpm] = useState('');
  const [simpleExactKeyscale, setSimpleExactKeyscale] = useState('');
  const [simpleExactTimesignature, setSimpleExactTimesignature] = useState('');

  const [caption, setCaption] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [lyricsDescription, setLyricsDescription] = useState('');
  const [lyricsGenre, setLyricsGenre] = useState('Pop');
  const [lyricsMood, setLyricsMood] = useState('Neutral');
  const [lyricsLanguage, setLyricsLanguage] = useState('English');
  const [bpm, setBpm] = useState('');
  const [keyScale, setKeyScale] = useState('');
  const [timeSignature, setTimeSignature] = useState('');
  const [duration, setDuration] = useState('60');
  const [vocalLanguage, setVocalLanguage] = useState('en');
  const [instrumental, setInstrumental] = useState(false);
  const [thinking, setThinking] = useState(true);
  const [inferenceSteps, setInferenceSteps] = useState('8');
  const [seed, setSeed] = useState('-1');
  const [audioFormat, setAudioFormat] = useState<'mp3' | 'wav' | 'flac'>('mp3');
  const [batchSize, setBatchSize] = useState('1');
  const [coverMode, setCoverMode] = useState(false);
  const [coverReferenceFile, setCoverReferenceFile] = useState<File | null>(null);
  const [audioCoverStrength, setAudioCoverStrength] = useState('0.6');

  const [healthRunning, setHealthRunning] = useState(false);
  const [healthAvailable, setHealthAvailable] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<'running' | 'succeeded' | 'failed' | string>('running');
  const [resultUrls, setResultUrls] = useState<string[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);
  const [presetName, setPresetName] = useState('');
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [presets, setPresets] = useState<MusicPreset[]>([]);
  const [historyItems, setHistoryItems] = useState<MusicHistoryItem[]>([]);

  const fullAudioUrls = useMemo(
    () => resultUrls.map((url) => `${settings.apiEndpoint}${url}`),
    [resultUrls, settings.apiEndpoint]
  );
  const isGenerating = loading || polling || (taskId !== null && taskStatus === 'running');

  useEffect(() => {
    const loadHealth = async () => {
      const health = await checkMusicHealth();
      if (health) {
        setHealthAvailable(health.available);
        setHealthRunning(health.running);
      }
    };
    loadHealth();
  }, [checkMusicHealth]);

  useEffect(() => {
    const loadLibrary = async () => {
      const presetResp = await listMusicPresets();
      if (presetResp) {
        setPresets(presetResp.presets);
      }
      const historyResp = await listMusicHistory(40);
      if (historyResp) {
        setHistoryItems(historyResp.history);
      }
    };
    loadLibrary();
  }, [listMusicPresets, listMusicHistory]);

  useEffect(() => {
    if (!taskId || taskStatus !== 'running') {
      setPolling(false);
      return;
    }

    setPolling(true);
    const interval = window.setInterval(async () => {
      const statusResult = await getMusicStatus(taskId);
      if (!statusResult) return;
      setTaskStatus(statusResult.status);
      if (statusResult.status === 'succeeded') {
        setResultUrls(statusResult.audios || []);
        setSuccessMessage('Music generation completed successfully.');
        setPolling(false);
        const health = await checkMusicHealth();
        if (health) setHealthRunning(health.running);
        const historyResp = await listMusicHistory(40);
        if (historyResp) setHistoryItems(historyResp.history);
      } else if (statusResult.status === 'failed') {
        setPolling(false);
        const historyResp = await listMusicHistory(40);
        if (historyResp) setHistoryItems(historyResp.history);
      }
    }, 2500);

    return () => window.clearInterval(interval);
  }, [taskId, taskStatus, getMusicStatus, checkMusicHealth, listMusicHistory]);

  const resetTaskState = () => {
    setTaskId(null);
    setTaskStatus('running');
    setResultUrls([]);
    setSuccessMessage(null);
  };

  const handleSimpleGenerate = async () => {
    resetTaskState();
    const normalizedMode: SimpleInputMode = simpleInputMode === 'exact' ? 'exact' : 'refine';
    const response = await simpleGenerateMusic({
      description: simpleDescription.trim(),
      input_mode: normalizedMode,
      instrumental: simpleInstrumental,
      vocal_language: simpleInstrumental ? undefined : simpleLanguage,
      duration: Number(simpleDuration),
      batch_size: Number(simpleBatchSize),
      exact_caption: normalizedMode === 'exact' ? simpleExactCaption.trim() || undefined : undefined,
      exact_lyrics:
        normalizedMode === 'exact' && !simpleInstrumental ? simpleExactLyrics.trim() || undefined : undefined,
      exact_bpm: normalizedMode === 'exact' && simpleExactBpm.trim() ? Number(simpleExactBpm) : undefined,
      exact_keyscale: normalizedMode === 'exact' ? simpleExactKeyscale.trim() || undefined : undefined,
      exact_timesignature:
        normalizedMode === 'exact' ? simpleExactTimesignature.trim() || undefined : undefined,
    });
    if (response?.task_id) {
      setTaskId(response.task_id);
      setTaskStatus('running');
      setSuccessMessage('Task submitted. Generating music...');
    }
  };

  const handleGenerateLyrics = async () => {
    const response = await generateLyrics({
      description: lyricsDescription.trim(),
      genre: lyricsGenre,
      mood: lyricsMood,
      language: lyricsLanguage,
      duration_hint: `${duration}s`,
    });
    if (response) {
      setLyrics(response.lyrics);
      if (response.caption) {
        setCaption(response.caption);
      }
      setSuccessMessage('Lyrics generated.');
    }
  };

  const handleCustomGenerate = async () => {
    resetTaskState();
    let response = null;
    if (coverMode) {
      if (!coverReferenceFile) {
        setSuccessMessage('Cover mode requires a reference audio file.');
        return;
      }
      response = await generateMusicCover(coverReferenceFile, {
        prompt: caption.trim(),
        lyrics: lyrics.trim(),
        duration: duration.trim() ? Number(duration) : undefined,
        audio_cover_strength: Number(audioCoverStrength),
        vocal_language: instrumental ? undefined : vocalLanguage,
        instrumental,
        thinking,
        inference_steps: Number(inferenceSteps),
        batch_size: Number(batchSize),
        seed: Number(seed),
        audio_format: audioFormat,
      });
    } else {
      response = await generateMusic({
        caption: caption.trim(),
        lyrics: lyrics.trim(),
        bpm: bpm.trim() ? Number(bpm) : undefined,
        keyscale: keyScale,
        timesignature: timeSignature,
        duration: duration.trim() ? Number(duration) : undefined,
        vocal_language: instrumental ? undefined : vocalLanguage,
        instrumental,
        thinking,
        inference_steps: Number(inferenceSteps),
        batch_size: Number(batchSize),
        seed: Number(seed),
        audio_format: audioFormat,
      });
    }
    if (response?.task_id) {
      setTaskId(response.task_id);
      setTaskStatus('running');
      setSuccessMessage('Task submitted. Generating music...');
    }
  };

  const handleDownload = async (audioUrl: string) => {
    const filename = audioUrl.split('/').pop();
    if (!filename) return;
    setDownloadingFile(filename);
    const blob = await downloadMusic(filename);
    setDownloadingFile(null);
    if (!blob) return;
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const applySimpleValues = (values: Record<string, unknown>) => {
    setSimpleDescription(String(values.description ?? ''));
    setSimpleInputMode(values.input_mode === 'exact' ? 'exact' : 'refine');
    setSimpleInstrumental(Boolean(values.instrumental ?? false));
    setSimpleLanguage(String(values.vocal_language ?? 'en'));
    setSimpleDuration(String(values.duration ?? '60'));
    setSimpleBatchSize(String(values.batch_size ?? '1'));
    setSimpleExactCaption(String(values.exact_caption ?? ''));
    setSimpleExactLyrics(String(values.exact_lyrics ?? ''));
    setSimpleExactBpm(values.exact_bpm == null ? '' : String(values.exact_bpm));
    setSimpleExactKeyscale(String(values.exact_keyscale ?? ''));
    setSimpleExactTimesignature(String(values.exact_timesignature ?? ''));
    setActiveTab('simple');
  };

  const applyCustomValues = (values: Record<string, unknown>) => {
    setCaption(String(values.caption ?? values.prompt ?? ''));
    setLyrics(String(values.lyrics ?? ''));
    setBpm(values.bpm == null ? '' : String(values.bpm));
    setKeyScale(String(values.keyscale ?? ''));
    setTimeSignature(String(values.timesignature ?? ''));
    setDuration(values.duration == null ? '60' : String(values.duration));
    setVocalLanguage(String(values.vocal_language ?? 'en'));
    setInstrumental(Boolean(values.instrumental ?? false));
    setThinking(values.thinking == null ? true : Boolean(values.thinking));
    setInferenceSteps(String(values.inference_steps ?? '8'));
    setSeed(String(values.seed ?? '-1'));
    setAudioFormat((values.audio_format as 'mp3' | 'wav' | 'flac') ?? 'mp3');
    setBatchSize(String(values.batch_size ?? '1'));
    setCoverMode(Boolean(values.cover_mode ?? (values.audio_cover_strength != null)));
    setAudioCoverStrength(String(values.audio_cover_strength ?? '0.6'));
    setCoverReferenceFile(null);
    setActiveTab('custom');
  };

  const handleApplyPreset = () => {
    const preset = presets.find((p) => p.id === selectedPresetId);
    if (!preset) return;
    if (preset.mode === 'simple') {
      applySimpleValues(preset.values);
    } else {
      applyCustomValues(preset.values);
    }
    setSuccessMessage(`Loaded preset: ${preset.name}`);
  };

  const handleSavePreset = async () => {
    const name = presetName.trim();
    if (!name) return;
    const values: Record<string, unknown> =
      activeTab === 'simple'
        ? {
            description: simpleDescription,
            input_mode: simpleInputMode,
            instrumental: simpleInstrumental,
            vocal_language: simpleLanguage,
            duration: Number(simpleDuration),
            batch_size: Number(simpleBatchSize),
            exact_caption: simpleInputMode === 'exact' ? simpleExactCaption : undefined,
            exact_lyrics: simpleInputMode === 'exact' ? simpleExactLyrics : undefined,
            exact_bpm: simpleInputMode === 'exact' && simpleExactBpm.trim() ? Number(simpleExactBpm) : undefined,
            exact_keyscale: simpleInputMode === 'exact' ? simpleExactKeyscale : undefined,
            exact_timesignature: simpleInputMode === 'exact' ? simpleExactTimesignature : undefined,
          }
        : {
            caption,
            lyrics,
            bpm: bpm.trim() ? Number(bpm) : undefined,
            keyscale: keyScale,
            timesignature: timeSignature,
            duration: duration.trim() ? Number(duration) : undefined,
            vocal_language: vocalLanguage,
            instrumental,
            thinking,
            inference_steps: Number(inferenceSteps),
            batch_size: Number(batchSize),
            seed: Number(seed),
            audio_format: audioFormat,
            cover_mode: coverMode,
            audio_cover_strength: coverMode ? Number(audioCoverStrength) : undefined,
          };

    const existing = presets.find((p) => p.id === selectedPresetId);
    if (existing) {
      const updated = await updateMusicPreset(existing.id, {
        name,
        mode: activeTab,
        values,
      });
      if (updated) {
        setSuccessMessage(`Updated preset: ${name}`);
      }
    } else {
      const created = await createMusicPreset({
        name,
        mode: activeTab,
        values,
      });
      if (created) {
        setSelectedPresetId(created.id);
        setSuccessMessage(`Saved preset: ${name}`);
      }
    }
    const presetResp = await listMusicPresets();
    if (presetResp) setPresets(presetResp.presets);
  };

  const handleDeletePreset = async () => {
    if (!selectedPresetId) return;
    const ok = await deleteMusicPreset(selectedPresetId);
    if (!ok) return;
    setSelectedPresetId('');
    setPresetName('');
    const presetResp = await listMusicPresets();
    if (presetResp) setPresets(presetResp.presets);
  };

  const handleLoadHistoryItem = (item: MusicHistoryItem) => {
    if (item.mode === 'simple') {
      applySimpleValues(item.request_payload);
    } else {
      applyCustomValues(item.request_payload);
    }
    setSuccessMessage(`Loaded settings from history task ${item.task_id}`);
  };

  const handleDeleteHistoryItem = async (historyId: string) => {
    const ok = await deleteMusicHistoryItem(historyId);
    if (!ok) return;
    const historyResp = await listMusicHistory(40);
    if (historyResp) setHistoryItems(historyResp.history);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Music Generation</h1>
          <p className="mt-2 text-gray-600">Generate music with ACE-Step using prompts, lyrics, and optional LLM assistance</p>
        </div>
        <div className={`px-3 py-2 rounded-md text-sm font-medium ${healthAvailable ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
          {healthAvailable ? (healthRunning ? 'ACE-Step Running' : 'ACE-Step Ready') : 'ACE-Step Not Configured'}
        </div>
      </div>

      {error && <Alert type="error" message={error} />}
      {successMessage && <Alert type="success" message={successMessage} />}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div className="p-4 border rounded-lg bg-gray-50 space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Presets</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Select
              label="Saved Presets"
              options={[
                { value: '', label: 'Select preset...' },
                ...presets.map((p) => ({
                  value: p.id,
                  label: `${p.name} (${p.mode})`,
                })),
              ]}
              value={selectedPresetId}
              onChange={(e) => {
                const id = e.target.value;
                setSelectedPresetId(id);
                const p = presets.find((x) => x.id === id);
                setPresetName(p?.name || '');
              }}
            />
            <Input
              label="Preset Name"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="e.g. CinematicBalladV1"
            />
            <div className="flex items-end gap-2">
              <Button variant="secondary" onClick={handleApplyPreset} disabled={!selectedPresetId}>
                Load
              </Button>
              <Button variant="secondary" onClick={handleSavePreset} disabled={!presetName.trim()}>
                Save
              </Button>
              <Button variant="danger" onClick={handleDeletePreset} disabled={!selectedPresetId}>
                Delete
              </Button>
            </div>
          </div>
        </div>

        <div className="flex gap-3">
          <Button variant={activeTab === 'simple' ? 'primary' : 'secondary'} onClick={() => setActiveTab('simple')}>
            Simple Mode
          </Button>
          <Button variant={activeTab === 'custom' ? 'primary' : 'secondary'} onClick={() => setActiveTab('custom')}>
            Custom Mode
          </Button>
        </div>

        {activeTab === 'simple' ? (
          <div className="space-y-4">
            <Select
              label="Simple Input Mode"
              options={[
                { value: 'refine', label: 'Refine with Ollama' },
                { value: 'exact', label: 'Exact ACE-Step Input' },
              ]}
              value={simpleInputMode}
              onChange={(e) => setSimpleInputMode(e.target.value as SimpleInputMode)}
            />
            <Input
              label={simpleInputMode === 'refine' ? 'Description' : 'Description / Context'}
              multiline
              rows={4}
              value={simpleDescription}
              onChange={(e) => setSimpleDescription(e.target.value)}
              placeholder={
                simpleInputMode === 'refine'
                  ? 'e.g. a chill lo-fi beat for studying with warm vinyl texture'
                  : 'Optional extra context for exact mode'
              }
              required
            />
            {simpleInputMode === 'exact' && (
              <div className="space-y-4 p-4 border rounded-lg bg-gray-50">
                <Input
                  label="Exact Caption"
                  value={simpleExactCaption}
                  onChange={(e) => setSimpleExactCaption(e.target.value)}
                  placeholder="ACE-Step caption/style prompt"
                />
                <Input
                  label="Exact Lyrics"
                  multiline
                  rows={6}
                  value={simpleExactLyrics}
                  onChange={(e) => setSimpleExactLyrics(e.target.value)}
                  placeholder="[Verse 1]..."
                  disabled={simpleInstrumental}
                />
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Input
                    label="Exact BPM (optional)"
                    value={simpleExactBpm}
                    onChange={(e) => setSimpleExactBpm(e.target.value)}
                  />
                  <Input
                    label="Exact Key/Scale (optional)"
                    value={simpleExactKeyscale}
                    onChange={(e) => setSimpleExactKeyscale(e.target.value)}
                  />
                  <Input
                    label="Exact Time Signature (optional)"
                    value={simpleExactTimesignature}
                    onChange={(e) => setSimpleExactTimesignature(e.target.value)}
                  />
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Select label="Vocal Language" options={LANGUAGE_OPTIONS} value={simpleLanguage} onChange={(e) => setSimpleLanguage(e.target.value)} disabled={simpleInstrumental} />
              <Input label="Duration (seconds)" type="text" value={simpleDuration} onChange={(e) => setSimpleDuration(e.target.value)} />
              <Select label="Batch Size" options={[{ value: '1', label: '1' }, { value: '2', label: '2' }, { value: '3', label: '3' }, { value: '4', label: '4' }]} value={simpleBatchSize} onChange={(e) => setSimpleBatchSize(e.target.value)} />
            </div>
            <label className="inline-flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={simpleInstrumental} onChange={(e) => setSimpleInstrumental(e.target.checked)} />
              Instrumental
            </label>
            <Button
              variant="primary"
              onClick={handleSimpleGenerate}
              isLoading={loading}
              disabled={
                isGenerating ||
                (simpleInputMode === 'refine' && !simpleDescription.trim()) ||
                (simpleInputMode === 'exact' &&
                  !simpleDescription.trim() &&
                  !simpleExactCaption.trim() &&
                  !simpleExactLyrics.trim())
              }
              className="w-full"
            >
              Generate
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <Input label="Caption / Style Prompt" value={caption} onChange={(e) => setCaption(e.target.value)} placeholder="e.g. cinematic pop ballad with emotional female vocal, piano, strings" />
            <Input label="Lyrics" multiline rows={10} value={lyrics} onChange={(e) => setLyrics(e.target.value)} placeholder="[Verse 1]..." />

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
              <Input label="Lyrics Idea" value={lyricsDescription} onChange={(e) => setLyricsDescription(e.target.value)} placeholder="Describe your song idea" />
              <Select label="Genre" options={GENRES} value={lyricsGenre} onChange={(e) => setLyricsGenre(e.target.value)} />
              <Select label="Mood" options={MOODS} value={lyricsMood} onChange={(e) => setLyricsMood(e.target.value)} />
              <Select label="Language" options={LANGUAGE_OPTIONS.map((x) => ({ value: x.label, label: x.label }))} value={lyricsLanguage} onChange={(e) => setLyricsLanguage(e.target.value)} />
            </div>
            <Button variant="secondary" onClick={handleGenerateLyrics} isLoading={loading} disabled={!lyricsDescription.trim()}>
              Generate Lyrics
            </Button>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Input label="BPM (Auto if blank)" value={bpm} onChange={(e) => setBpm(e.target.value)} />
              <Select label="Key / Scale" options={KEY_OPTIONS} value={keyScale} onChange={(e) => setKeyScale(e.target.value)} />
              <Select label="Time Signature" options={TIME_SIGNATURE_OPTIONS} value={timeSignature} onChange={(e) => setTimeSignature(e.target.value)} />
              <Input label="Duration (seconds)" value={duration} onChange={(e) => setDuration(e.target.value)} />
              <Select label="Vocal Language" options={LANGUAGE_OPTIONS} value={vocalLanguage} onChange={(e) => setVocalLanguage(e.target.value)} disabled={instrumental} />
              <Select label="Batch Size" options={[{ value: '1', label: '1' }, { value: '2', label: '2' }, { value: '3', label: '3' }, { value: '4', label: '4' }]} value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
            </div>

            <div className="flex gap-6">
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={instrumental} onChange={(e) => setInstrumental(e.target.checked)} />
                Instrumental
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
                Thinking (LLM-assisted)
              </label>
            </div>

            <Button variant="secondary" onClick={() => setShowAdvanced((prev) => !prev)}>
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </Button>
            {showAdvanced && (
              <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
                <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={coverMode}
                    onChange={(e) => {
                      setCoverMode(e.target.checked);
                      if (!e.target.checked) {
                        setCoverReferenceFile(null);
                      }
                    }}
                  />
                  Cover Mode (Reference Audio)
                </label>

                {coverMode && (
                  <div className="space-y-3 border rounded-md p-3 bg-white">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Reference Audio</label>
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={(e) => setCoverReferenceFile(e.target.files?.[0] || null)}
                        className="mt-1 block w-full text-sm text-gray-700"
                      />
                      {coverReferenceFile && (
                        <div className="mt-1 flex items-center justify-between gap-2 text-xs text-gray-600">
                          <span>{coverReferenceFile.name}</span>
                          <button
                            type="button"
                            className="text-primary-700 underline hover:text-primary-800"
                            onClick={() => setCoverReferenceFile(null)}
                          >
                            Remove
                          </button>
                        </div>
                      )}
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700">
                        Cover Strength: {Number(audioCoverStrength).toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={audioCoverStrength}
                        onChange={(e) => setAudioCoverStrength(e.target.value)}
                        className="mt-1 w-full"
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        0.8-1.0 tight structure copy, 0.4-0.6 groove + freedom, 0.0-0.2 loose style hint.
                      </p>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Input label="Inference Steps" value={inferenceSteps} onChange={(e) => setInferenceSteps(e.target.value)} />
                  <Input label="Seed (-1 random)" value={seed} onChange={(e) => setSeed(e.target.value)} />
                  <Select label="Audio Format" options={AUDIO_FORMAT_OPTIONS} value={audioFormat} onChange={(e) => setAudioFormat(e.target.value as 'mp3' | 'wav' | 'flac')} />
                </div>
              </div>
            )}

            <Button
              variant="primary"
              onClick={handleCustomGenerate}
              isLoading={loading}
              disabled={(!caption.trim() && !lyrics.trim()) || isGenerating || (coverMode && !coverReferenceFile)}
              className="w-full"
            >
              Generate Music
            </Button>
          </div>
        )}
      </div>

      {(polling || (taskId && taskStatus === 'running')) && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center gap-3 text-gray-700">
            <LoadingSpinner size="sm" />
            <span>Generating music for task `{taskId}`...</span>
          </div>
        </div>
      )}

      {taskStatus === 'failed' && (
        <Alert type="error" message={`Music generation failed for task ${taskId || ''}.`} />
      )}

      {fullAudioUrls.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">Generated Music</h2>
          {fullAudioUrls.map((audioUrl, idx) => {
            const relativeUrl = resultUrls[idx];
            const filename = relativeUrl?.split('/').pop() || `track-${idx + 1}.mp3`;
            return (
              <div key={`${audioUrl}-${idx}`} className="space-y-2 border rounded-md p-4">
                <AudioPlayer src={audioUrl} filename={filename} />
                <Button
                  variant="secondary"
                  onClick={() => handleDownload(relativeUrl)}
                  isLoading={downloadingFile === filename}
                  className="w-full"
                >
                  Download
                </Button>
              </div>
            );
          })}
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">Generation History</h2>
          <Button
            variant="secondary"
            onClick={async () => {
              const historyResp = await listMusicHistory(40);
              if (historyResp) setHistoryItems(historyResp.history);
            }}
          >
            Refresh
          </Button>
        </div>
        {historyItems.length === 0 ? (
          <p className="text-sm text-gray-600">No history yet.</p>
        ) : (
          <div className="space-y-3">
            {historyItems.map((item) => (
              <div key={item.id} className="border rounded-md p-4 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-gray-700">
                    <div><span className="font-medium">Task:</span> {item.task_id}</div>
                    <div><span className="font-medium">Mode:</span> {item.mode}</div>
                    <div><span className="font-medium">Status:</span> {item.status}</div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="secondary" onClick={() => handleLoadHistoryItem(item)}>
                      Load Settings
                    </Button>
                    <Button variant="danger" onClick={() => handleDeleteHistoryItem(item.id)}>
                      Delete
                    </Button>
                  </div>
                </div>
                {item.audios?.length > 0 && (
                  <div className="space-y-2">
                    {item.audios.map((relUrl, idx) => {
                      const filename = relUrl.split('/').pop() || `history-${idx + 1}.mp3`;
                      return (
                        <div key={`${item.id}-${idx}`} className="space-y-2 border rounded-md p-3">
                          <AudioPlayer src={`${settings.apiEndpoint}${relUrl}`} filename={filename} />
                          <button
                            type="button"
                            className="text-sm font-medium text-primary-700 underline hover:text-primary-800 disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={() => handleDownload(relUrl)}
                            disabled={downloadingFile === filename}
                          >
                            {downloadingFile === filename ? 'Downloading...' : 'Download'}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
