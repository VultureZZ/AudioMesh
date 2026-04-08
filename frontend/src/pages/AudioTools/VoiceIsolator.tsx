import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  createVoiceFromClip,
  getSpeakerIsolationStatus,
  uploadForSpeakerIsolation,
} from '../../api/audioToolsApi';
import type { SpeakerIsolationClipItem, SpeakerIsolationStatusResponse } from '../../types/api';
import { Alert, ToastContainer } from '../../components/Alert';
import { Button } from '../../components/Button';
import { Input } from '../../components/Input';
import { LoadingSpinner } from '../../components/LoadingSpinner';
import { useSettings } from '../../hooks/useSettings';
import { isValidAudioFile, validateVoiceName } from '../../utils/validation';

type Toast = { id: string; type: 'success' | 'error' | 'info' | 'warning'; message: string };

/** Distinct card accents per speaker slot (1–6); adjust Tailwind classes here. */
const SPEAKER_CARD_ACCENTS = [
  { border: 'border-indigo-400', bg: 'bg-indigo-50/90', chip: 'bg-indigo-500', heading: 'text-indigo-950' },
  { border: 'border-teal-400', bg: 'bg-teal-50/90', chip: 'bg-teal-500', heading: 'text-teal-950' },
  { border: 'border-amber-400', bg: 'bg-amber-50/90', chip: 'bg-amber-500', heading: 'text-amber-950' },
  { border: 'border-rose-400', bg: 'bg-rose-50/90', chip: 'bg-rose-500', heading: 'text-rose-950' },
  { border: 'border-violet-400', bg: 'bg-violet-50/90', chip: 'bg-violet-500', heading: 'text-violet-950' },
  { border: 'border-cyan-400', bg: 'bg-cyan-50/90', chip: 'bg-cyan-500', heading: 'text-cyan-950' },
] as const;

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const s = Math.floor(seconds % 60);
  const m = Math.floor((seconds / 60) % 60);
  const h = Math.floor(seconds / 3600);
  const ss = s.toString().padStart(2, '0');
  const mm = m.toString().padStart(2, '0');
  if (h > 0) return `${h}:${mm}:${ss}`;
  return `${m}:${ss}`;
}

function clipStreamUrl(apiEndpoint: string, downloadUrl: string, apiKey?: string): string {
  let u = `${apiEndpoint}${downloadUrl}`;
  if (apiKey?.trim()) {
    u += `${downloadUrl.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(apiKey)}`;
  }
  return u;
}

function phaseLabel(submitting: boolean, status: SpeakerIsolationStatusResponse | null): string {
  if (submitting) return 'Uploading…';
  if (!status) return '';
  if (status.status === 'failed') return status.error || 'Failed';
  if (status.status === 'complete') return 'Complete';
  if (status.status === 'queued') return 'Running diarization…';
  if (status.status === 'diarizing') return 'Running diarization…';
  if (status.status === 'extracting') return 'Extracting clips…';
  if (status.status === 'inferring_names') return 'Inferring speaker names…';
  return status.current_stage || status.status;
}

function progressValue(submitting: boolean, uploadPct: number, status: SpeakerIsolationStatusResponse | null): number {
  if (submitting) return uploadPct;
  return status?.progress_pct ?? 0;
}

export function VoiceIsolatorPage() {
  const { settings } = useSettings();
  const [file, setFile] = useState<File | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<SpeakerIsolationStatusResponse | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const pollIntervalRef = useRef<number | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [modalClip, setModalClip] = useState<SpeakerIsolationClipItem | null>(null);
  const [modalSpeakerLabel, setModalSpeakerLabel] = useState('');
  const [voiceName, setVoiceName] = useState('');
  const [voiceDescription, setVoiceDescription] = useState('');
  const [creatingVoice, setCreatingVoice] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  const pushToast = useCallback((type: Toast['type'], message: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setToasts((t) => [...t, { id, type, message }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const onFileChosen = useCallback((f: File | null) => {
    setFile(f);
    setJobId(null);
    setStatus(null);
    setUploadPct(0);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) void onFileChosen(f);
    },
    [onFileChosen]
  );

  const processing = Boolean(
    jobId && status && !['complete', 'failed'].includes(status.status)
  );

  useEffect(() => {
    if (!jobId) return;
    const tick = async () => {
      try {
        const s = await getSpeakerIsolationStatus(jobId);
        setStatus(s);
        if (s.status === 'complete' || s.status === 'failed') {
          if (pollIntervalRef.current !== null) {
            window.clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
        }
      } catch (e) {
        pushToast('error', e instanceof Error ? e.message : 'Status poll failed');
      }
    };
    void tick();
    pollIntervalRef.current = window.setInterval(() => void tick(), 2000);
    return () => {
      if (pollIntervalRef.current !== null) {
        window.clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [jobId, pushToast]);

  const handleDetect = async () => {
    if (!file || !isValidAudioFile(file)) {
      pushToast('error', 'Choose a valid audio file (MP3, WAV, M4A, MP4).');
      return;
    }
    setSubmitting(true);
    setUploadPct(0);
    setStatus(null);
    setJobId(null);
    try {
      const res = await uploadForSpeakerIsolation(file, (p) => setUploadPct(p));
      setJobId(res.job_id);
      const s = await getSpeakerIsolationStatus(res.job_id);
      setStatus(s);
      pushToast('success', 'Upload complete. Detecting speakers…');
    } catch (e) {
      pushToast('error', e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setSubmitting(false);
    }
  };

  const openCreateModal = (clip: SpeakerIsolationClipItem, speakerLabel: string) => {
    setModalClip(clip);
    setModalSpeakerLabel(speakerLabel);
    setVoiceName('');
    setVoiceDescription('');
    setModalError(null);
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setModalClip(null);
    setModalError(null);
  };

  const submitCreateVoice = async () => {
    if (!jobId || !modalClip) return;
    const nv = validateVoiceName(voiceName);
    if (!nv.valid) {
      setModalError(nv.error || 'Invalid name');
      return;
    }
    setCreatingVoice(true);
    setModalError(null);
    try {
      const res = await createVoiceFromClip(jobId, modalClip.clip_id, voiceName.trim(), voiceDescription.trim() || undefined);
      if (res.voice) {
        pushToast('success', `Voice "${res.voice.name}" was created.`);
      } else {
        pushToast('success', res.message || 'Voice created.');
      }
      closeModal();
    } catch (e) {
      setModalError(e instanceof Error ? e.message : 'Could not create voice');
    } finally {
      setCreatingVoice(false);
    }
  };

  const modalPreviewSrc = useMemo(() => {
    if (!modalClip) return '';
    return clipStreamUrl(settings.apiEndpoint, modalClip.download_url, settings.apiKey);
  }, [modalClip, settings.apiEndpoint, settings.apiKey]);

  const barPct = progressValue(submitting, uploadPct, status);

  return (
    <div className="max-w-5xl">
      <ToastContainer toasts={toasts} onRemove={removeToast} />

      <h1 className="text-2xl font-semibold text-gray-900">Speaker Voice Isolator</h1>
      <p className="mt-1 text-sm text-gray-600">
        Upload mixed audio. We run the same speaker diarization as transcripts, pick up to three strong 10–15 second
        clips per speaker (up to six speakers), preview each clip, and clone a voice from any clip.
      </p>

      <div
        className="mt-6 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-white hover:border-primary-400 transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept=".mp3,.wav,.m4a,.mp4,audio/mpeg,audio/wav,audio/x-m4a,audio/mp4"
          className="hidden"
          id="voice-isolator-file"
          onChange={(e) => void onFileChosen(e.target.files?.[0] ?? null)}
        />
        <label htmlFor="voice-isolator-file" className="cursor-pointer text-primary-600 font-medium">
          Click to browse
        </label>
        <span className="text-gray-600"> or drag and drop MP3, WAV, M4A, or MP4 (max 500MB)</span>
        {file && (
          <div className="mt-4 text-left max-w-lg mx-auto space-y-2">
            <div className="text-sm text-gray-800">
              <span className="font-medium">File:</span> {file.name}
            </div>
            {submitting && (
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Upload</span>
                  <span>{uploadPct}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 transition-all duration-300"
                    style={{ width: `${uploadPct}%` }}
                  />
                </div>
              </div>
            )}
            <button
              type="button"
              onClick={() => void handleDetect()}
              disabled={submitting || processing}
              className="mt-2 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50"
            >
              {submitting ? (
                <>
                  <span className="mr-2 inline-flex">
                    <LoadingSpinner size="sm" />
                  </span>
                  Uploading…
                </>
              ) : (
                'Detect Speakers'
              )}
            </button>
          </div>
        )}
      </div>

      {jobId && (
        <div className="mt-8 rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-medium text-gray-900">Processing</h2>
          {!status ? (
            <div className="mt-4 space-y-3 animate-pulse">
              <div className="h-3 bg-gray-200 rounded w-full" />
              <div className="h-3 bg-gray-200 rounded w-3/4" />
            </div>
          ) : (
            <>
              <p className="mt-2 text-sm text-gray-600">{phaseLabel(submitting, status)}</p>
              <div className="mt-3">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Progress</span>
                  <span>{barPct}%</span>
                </div>
                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 transition-all duration-500 ease-out"
                    style={{ width: `${barPct}%` }}
                  />
                </div>
              </div>
              {status.status === 'failed' && (
                <div className="mt-4">
                  <Alert type="error" message={status.error || 'Processing failed'} />
                </div>
              )}
            </>
          )}
        </div>
      )}

      {status?.status === 'complete' && status.speakers && status.speakers.length > 0 && (
        <div className="mt-8">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Speakers &amp; clips</h2>
          <div className="grid gap-6 sm:grid-cols-2">
            {status.speakers.map((sp, si) => {
              const acc = SPEAKER_CARD_ACCENTS[si % SPEAKER_CARD_ACCENTS.length];
              return (
                <div
                  key={sp.speaker_id}
                  className={`rounded-xl border-2 shadow-sm p-4 ${acc.border} ${acc.bg}`}
                >
                  <div className="flex items-start gap-3">
                    <span
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-white text-sm font-semibold ${acc.chip}`}
                      aria-hidden
                    >
                      {si + 1}
                    </span>
                    <div>
                      <h3 className={`text-base font-semibold ${acc.heading}`}>{sp.label}</h3>
                      {sp.label_source === 'inferred' && (
                        <p className="text-xs text-gray-500 mt-0.5">Name detected from speech</p>
                      )}
                      <p className="text-xs text-gray-600 mt-0.5">
                        Total speaking time: {formatTime(sp.total_speaking_seconds)}
                      </p>
                    </div>
                  </div>
                  <ul className="mt-4 space-y-4">
                    {sp.clips.map((clip, ci) => (
                      <li key={clip.clip_id} className="rounded-lg bg-white/80 border border-gray-200/80 p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-sm font-medium text-gray-800">
                            Clip {ci + 1} — {Math.round(clip.duration_seconds)}s
                          </span>
                          <button
                            type="button"
                            onClick={() => openCreateModal(clip, sp.label)}
                            className="text-sm font-medium text-primary-700 hover:text-primary-900"
                          >
                            Create Voice from This Clip
                          </button>
                        </div>
                        <audio
                          controls
                          preload="none"
                          className="mt-2 w-full max-w-full"
                          src={clipStreamUrl(settings.apiEndpoint, clip.download_url, settings.apiKey)}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {status?.status === 'complete' && (!status.speakers || status.speakers.length === 0) && (
        <div className="mt-8 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          No speaker segments were extracted. Try a longer file with clear speech, and ensure diarization (HF_TOKEN) is
          configured on the server.
        </div>
      )}

      {modalOpen && modalClip && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="voice-isolator-modal-title"
        >
          <div className="w-full max-w-md rounded-lg bg-white shadow-xl border border-gray-200 p-6">
            <h2 id="voice-isolator-modal-title" className="text-lg font-semibold text-gray-900">
              Create voice from clip
            </h2>
            <p className="text-sm text-gray-600 mt-1">{modalSpeakerLabel}</p>

            <div className="mt-4 space-y-3">
              <label className="block text-sm font-medium text-gray-700">
                Voice name
                <Input
                  value={voiceName}
                  onChange={(e) => setVoiceName(e.target.value)}
                  placeholder="e.g. Podcast Host A"
                  className="mt-1"
                  autoComplete="off"
                />
              </label>
              <label className="block text-sm font-medium text-gray-700">
                Description (optional)
                <Input
                  value={voiceDescription}
                  onChange={(e) => setVoiceDescription(e.target.value)}
                  placeholder="Short note for your library"
                  className="mt-1"
                  autoComplete="off"
                />
              </label>
            </div>

            <p className="text-xs text-gray-500 mt-3">Preview (same clip as on the card)</p>
            <audio controls preload="none" className="mt-1 w-full" src={modalPreviewSrc} />

            {modalError && (
              <div className="mt-3">
                <Alert type="error" message={modalError} />
              </div>
            )}

            <div className="mt-6 flex flex-wrap gap-3 justify-end">
              <Button type="button" variant="secondary" onClick={closeModal}>
                Cancel
              </Button>
              <Button type="button" onClick={() => void submitCreateVoice()} isLoading={creatingVoice}>
                Create Voice
              </Button>
            </div>

            <p className="mt-4 text-sm text-gray-600">
              After creating, open the{' '}
              <Link to="/voices" className="text-primary-700 font-medium hover:underline">
                Voices
              </Link>{' '}
              page to use this voice for generation.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
