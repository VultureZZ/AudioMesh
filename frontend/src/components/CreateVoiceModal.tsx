/**
 * Unified modal to create a voice: from audio files, from clips, or from a text description (VoiceDesign).
 */
import { useMemo, useRef, useState } from 'react';
import type { AudioClipRange, VoiceCreateResponse } from '../types/api';
import { validateVoiceName, isValidAudioFile } from '../utils/validation';
import { Alert } from './Alert';
import { Button } from './Button';
import { Input } from './Input';
import { Select } from './Select';
import { FileUpload } from './FileUpload';
import { SUPPORTED_LANGUAGES } from '../utils/languages';

export type CreateVoiceSource = 'audio' | 'clips' | 'prompt';

interface CreateVoiceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (params: {
    name: string;
    description?: string;
    creation_source: CreateVoiceSource;
    audio_files?: File[];
    audio_file?: File;
    clip_ranges?: AudioClipRange[];
    voice_design_prompt?: string;
    keywords?: string;
    language_code?: string;
    gender?: string;
    image?: File;
  }) => Promise<VoiceCreateResponse | null>;
}

export function CreateVoiceModal({ isOpen, onClose, onCreate }: CreateVoiceModalProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [creationSource, setCreationSource] = useState<CreateVoiceSource>('audio');

  const [voiceName, setVoiceName] = useState('');
  const [voiceDescription, setVoiceDescription] = useState('');
  const [voiceKeywords, setVoiceKeywords] = useState('');
  const [voiceLanguageCode, setVoiceLanguageCode] = useState<string>('');
  const [voiceGender, setVoiceGender] = useState<string>('unknown');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [markEnd, setMarkEnd] = useState<number | null>(null);
  const [clips, setClips] = useState<AudioClipRange[]>([]);
  const [voiceDesignPrompt, setVoiceDesignPrompt] = useState('');

  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nameValidation = validateVoiceName(voiceName);
  const totalSelectedSeconds = useMemo(
    () => clips.reduce((sum, c) => sum + Math.max(0, c.end_seconds - c.start_seconds), 0),
    [clips]
  );

  const resetAndClose = () => {
    setCreationSource('audio');
    setVoiceName('');
    setVoiceDescription('');
    setVoiceKeywords('');
    setVoiceLanguageCode('');
    setVoiceGender('unknown');
    setSelectedImage(null);
    setSelectedFiles([]);
    setAudioFile(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setMarkStart(null);
    setMarkEnd(null);
    setClips([]);
    setVoiceDesignPrompt('');
    setError(null);
    onClose();
  };

  const handlePickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (!f) return;
    if (!isValidAudioFile(f)) {
      setError('Please choose a valid audio file');
      return;
    }
    setError(null);
    setAudioFile(f);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(f));
    setMarkStart(null);
    setMarkEnd(null);
    setClips([]);
  };

  const getCurrentTime = () => (audioRef.current?.currentTime ?? 0);

  const handleAddClip = () => {
    if (markStart === null || markEnd === null) {
      setError('Please mark both start and end before adding a clip');
      return;
    }
    const start = Math.min(markStart, markEnd);
    const end = Math.max(markStart, markEnd);
    if (end - start < 0.5) {
      setError('Clip is too short (minimum 0.5s)');
      return;
    }
    setError(null);
    setClips((prev) => [...prev, { start_seconds: start, end_seconds: end }]);
  };

  const handleCreate = async () => {
    if (!nameValidation.valid) {
      setError(nameValidation.error || 'Invalid voice name');
      return;
    }
    if (creationSource === 'audio') {
      if (selectedFiles.length === 0) {
        setError('Select at least one audio file');
        return;
      }
    } else if (creationSource === 'clips') {
      if (!audioFile || clips.length === 0) {
        setError('Select an audio file and add at least one clip');
        return;
      }
    } else {
      if (!voiceDesignPrompt.trim()) {
        setError('Enter a voice description');
        return;
      }
    }

    setCreating(true);
    setError(null);

    const base = {
      name: voiceName.trim(),
      description: voiceDescription.trim() || undefined,
      creation_source: creationSource,
      keywords: voiceKeywords.trim() || undefined,
      language_code: voiceLanguageCode || undefined,
      gender: voiceGender || undefined,
      image: selectedImage || undefined,
    };

    const params =
      creationSource === 'audio'
        ? { ...base, audio_files: selectedFiles }
        : creationSource === 'clips' && audioFile
          ? { ...base, audio_file: audioFile, clip_ranges: clips }
          : creationSource === 'prompt'
            ? { ...base, voice_design_prompt: voiceDesignPrompt.trim() }
            : base;

    const resp = await onCreate(params);
    setCreating(false);
    if (!resp) return;
    if (!resp.success) {
      setError(resp.message || 'Failed to create voice');
      return;
    }
    resetAndClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Create Voice</h2>
          <Button variant="secondary" onClick={resetAndClose} disabled={creating}>
            Close
          </Button>
        </div>

        {error && <Alert type="error" message={error} onClose={() => setError(null)} />}

        <div className="space-y-4 mb-4">
          <label className="block text-sm font-medium text-gray-700">Create from</label>
          <div className="flex gap-4">
            <label className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="creation_source"
                checked={creationSource === 'audio'}
                onChange={() => setCreationSource('audio')}
                className="rounded border-gray-300"
              />
              <span>Audio files</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="creation_source"
                checked={creationSource === 'clips'}
                onChange={() => setCreationSource('clips')}
                className="rounded border-gray-300"
              />
              <span>Clips from one file</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="creation_source"
                checked={creationSource === 'prompt'}
                onChange={() => setCreationSource('prompt')}
                className="rounded border-gray-300"
              />
              <span>Description (VoiceDesign)</span>
            </label>
          </div>
        </div>

        <div className="space-y-4">
          <Input
            label="Voice Name"
            value={voiceName}
            onChange={(e) => setVoiceName(e.target.value)}
            error={voiceName && !nameValidation.valid ? nameValidation.error : undefined}
            required
            placeholder="e.g., My Custom Voice"
          />
          <Input
            label="Description (Optional)"
            multiline
            rows={2}
            value={voiceDescription}
            onChange={(e) => setVoiceDescription(e.target.value)}
            placeholder="Describe this voice..."
          />
          <Input
            label="Keywords (Optional)"
            value={voiceKeywords}
            onChange={(e) => setVoiceKeywords(e.target.value)}
            placeholder="e.g., calm, technical (comma-separated)"
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Language (Optional)"
              options={[{ value: '', label: 'Unknown' }, { value: 'in', label: 'Indian' }, ...SUPPORTED_LANGUAGES]}
              value={voiceLanguageCode}
              onChange={(e) => setVoiceLanguageCode(e.target.value)}
            />
            <Select
              label="Gender (Optional)"
              options={[
                { value: 'unknown', label: 'Unknown' },
                { value: 'female', label: 'Female' },
                { value: 'male', label: 'Male' },
                { value: 'neutral', label: 'Gender-neutral' },
              ]}
              value={voiceGender}
              onChange={(e) => setVoiceGender(e.target.value)}
            />
          </div>

          {creationSource === 'audio' && (
            <div>
              <FileUpload
                onFilesChange={setSelectedFiles}
                error={selectedFiles.length === 0 ? 'Select at least one audio file' : undefined}
              />
              <p className="mt-1 text-xs text-gray-500">Combine multiple files. For best quality use 5-15 seconds total.</p>
            </div>
          )}

          {creationSource === 'clips' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">Audio file</label>
              <input type="file" accept="audio/*" onChange={handlePickFile} disabled={creating} />
              {audioFile && <p className="text-xs text-gray-600">{audioFile.name}</p>}
              {audioUrl && (
                <>
                  <audio ref={audioRef} src={audioUrl} controls className="w-full mt-2" />
                  <div className="flex flex-wrap gap-2 mt-2">
                    <Button variant="secondary" onClick={() => setMarkStart(getCurrentTime())} disabled={creating}>
                      Mark start ({markStart !== null ? markStart.toFixed(2) : '—'}s)
                    </Button>
                    <Button variant="secondary" onClick={() => setMarkEnd(getCurrentTime())} disabled={creating}>
                      Mark end ({markEnd !== null ? markEnd.toFixed(2) : '—'}s)
                    </Button>
                    <Button
                      variant="primary"
                      onClick={handleAddClip}
                      disabled={creating || markStart === null || markEnd === null}
                    >
                      Add clip
                    </Button>
                  </div>
                  <p className="text-xs text-gray-600">
                    Total selected: {totalSelectedSeconds.toFixed(2)}s (5-15s ideal, max 60s).
                  </p>
                </>
              )}
              {clips.length > 0 && (
                <div className="mt-2 space-y-1">
                  <span className="text-sm font-medium">Clips: {clips.length}</span>
                  {clips.map((c, i) => (
                    <div key={i} className="flex justify-between text-sm bg-gray-50 p-2 rounded">
                      <span>
                        {c.start_seconds.toFixed(2)}s – {c.end_seconds.toFixed(2)}s
                      </span>
                      <button
                        type="button"
                        onClick={() => setClips((p) => p.filter((_, j) => j !== i))}
                        className="text-red-600 hover:underline"
                        disabled={creating}
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {creationSource === 'prompt' && (
            <div>
              <Input
                label="Voice description"
                multiline
                rows={4}
                value={voiceDesignPrompt}
                onChange={(e) => setVoiceDesignPrompt(e.target.value)}
                placeholder="e.g., A clear female voice, moderate speed, stable tone, suitable for news or conversation. Or: Young male, warm and confident, slight rasp, calm delivery."
                required
              />
              <p className="mt-1 text-xs text-gray-500">
                Be specific and multi-dimensional. Include: Persona (male/female, age), Pace (fast/moderate/slow),
                Timbre (clear, deep, warm, raspy), Emotion (calm, enthusiastic, serious), and optionally Scenario (news, storytelling).
                Rich descriptions adhere better than single words.
              </p>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Avatar image (optional)</label>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-primary-50 file:text-primary-700"
              onChange={(e) => setSelectedImage(e.target.files?.[0] ?? null)}
            />
          </div>

          <Button
            variant="primary"
            onClick={handleCreate}
            isLoading={creating}
            disabled={
              !nameValidation.valid ||
              (creationSource === 'audio' && selectedFiles.length === 0) ||
              (creationSource === 'clips' && (!audioFile || clips.length === 0)) ||
              (creationSource === 'prompt' && !voiceDesignPrompt.trim())
            }
            className="w-full"
          >
            Create Voice
          </Button>
        </div>
      </div>
    </div>
  );
}
