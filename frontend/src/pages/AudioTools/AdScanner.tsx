import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  downloadAudioToolsExport,
  exportPodcastAdAudio,
  getPodcastAdScanStatus,
  scanPodcastAds,
} from '../../api/audioToolsApi';
import type { AdSegmentItem, PodcastAdScanStatusResponse } from '../../types/api';
import { Alert, ToastContainer } from '../../components/Alert';
import { LoadingSpinner } from '../../components/LoadingSpinner';

type Toast = { id: string; type: 'success' | 'error' | 'info' | 'warning'; message: string };

type TimelinePart =
  | { kind: 'podcast'; start: number; end: number }
  | { kind: 'ad'; start: number; end: number; label: string; confidence: number };

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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function buildTimeline(duration: number, ads: AdSegmentItem[]): TimelinePart[] {
  if (duration <= 0) return [];
  const sorted = [...ads].sort((a, b) => a.start_seconds - b.start_seconds);
  const parts: TimelinePart[] = [];
  let cursor = 0;
  for (const ad of sorted) {
    const a = Math.max(0, ad.start_seconds);
    const b = Math.min(duration, ad.end_seconds);
    if (b <= a) continue;
    if (a > cursor) {
      parts.push({ kind: 'podcast', start: cursor, end: a });
    }
    parts.push({
      kind: 'ad',
      start: a,
      end: b,
      label: ad.label,
      confidence: ad.confidence,
    });
    cursor = Math.max(cursor, b);
  }
  if (cursor < duration) {
    parts.push({ kind: 'podcast', start: cursor, end: duration });
  }
  return parts;
}

function readAudioDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.preload = 'metadata';
    const done = (v: number | null) => {
      URL.revokeObjectURL(url);
      resolve(v);
    };
    audio.onloadedmetadata = () => done(Number.isFinite(audio.duration) ? audio.duration : null);
    audio.onerror = () => done(null);
    audio.src = url;
  });
}

export function AdScannerPage() {
  const [file, setFile] = useState<File | null>(null);
  const [durationEstimate, setDurationEstimate] = useState<number | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<PodcastAdScanStatusResponse | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [exportClean, setExportClean] = useState<{ duration_seconds: number; file_size_bytes: number } | null>(
    null
  );
  const [exportAds, setExportAds] = useState<{ duration_seconds: number; file_size_bytes: number } | null>(null);
  const [exportingClean, setExportingClean] = useState(false);
  const [exportingAds, setExportingAds] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pollIntervalRef = useRef<number | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const pushToast = useCallback((type: Toast['type'], message: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setToasts((t) => [...t, { id, type, message }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const onFileChosen = useCallback(async (f: File | null) => {
    setFile(f);
    setDurationEstimate(null);
    setAudioUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (!f) return;
    const d = await readAudioDuration(f);
    setDurationEstimate(d);
    setAudioUrl(URL.createObjectURL(f));
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
    jobId &&
      status &&
      !['complete', 'failed'].includes(status.status)
  );

  useEffect(() => {
    if (!jobId) return;
    const tick = async () => {
      try {
        const s = await getPodcastAdScanStatus(jobId);
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

  const timeline = useMemo(() => {
    const d = status?.duration_seconds ?? durationEstimate ?? 0;
    const ads = status?.ad_segments ?? [];
    if (!d || status?.status !== 'complete') return [];
    return buildTimeline(d, ads);
  }, [status, durationEstimate]);

  const totalDuration = status?.duration_seconds ?? durationEstimate ?? 0;

  const handleScan = async () => {
    if (!file) return;
    setSubmitting(true);
    setUploadPct(0);
    setStatus(null);
    setJobId(null);
    setExportClean(null);
    setExportAds(null);
    try {
      const res = await scanPodcastAds(file, (p) => setUploadPct(p));
      setJobId(res.job_id);
      const s = await getPodcastAdScanStatus(res.job_id);
      setStatus(s);
      pushToast('success', 'Upload complete. Processing started.');
    } catch (e) {
      pushToast('error', e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setSubmitting(false);
    }
  };

  const seekTo = (t: number) => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, t);
    void el.play().catch(() => undefined);
  };

  const runExport = async (mode: 'clean' | 'ads_only') => {
    if (!jobId) return;
    const setBusy = mode === 'clean' ? setExportingClean : setExportingAds;
    setBusy(true);
    try {
      const res = await exportPodcastAdAudio(jobId, mode);
      const blob = await downloadAudioToolsExport(res.filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = res.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      const meta = { duration_seconds: res.duration_seconds, file_size_bytes: res.file_size_bytes };
      if (mode === 'clean') setExportClean(meta);
      else setExportAds(meta);
      pushToast('success', 'Export ready.');
    } catch (e) {
      pushToast('error', e instanceof Error ? e.message : 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  const stageLabel = (s: PodcastAdScanStatusResponse | null) => {
    if (!s) return '';
    if (s.status === 'failed') return s.error || 'Failed';
    if (s.status === 'complete') return 'Complete';
    return s.current_stage || s.status;
  };

  return (
    <div className="max-w-5xl">
      <ToastContainer toasts={toasts} onRemove={removeToast} />

      <h1 className="text-2xl font-semibold text-gray-900">Podcast Ad Scanner</h1>
      <p className="mt-1 text-sm text-gray-600">
        Upload a podcast episode. We transcribe it with Whisper, then use your configured Ollama model to find likely
        ad and sponsor segments. Export a clean cut or an ads-only clip as MP3.
      </p>

      <div
        className="mt-6 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-white hover:border-primary-400 transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept=".mp3,.wav,.m4a,audio/mpeg,audio/wav,audio/x-m4a,audio/mp4"
          className="hidden"
          id="ad-scan-file"
          onChange={(e) => void onFileChosen(e.target.files?.[0] ?? null)}
        />
        <label htmlFor="ad-scan-file" className="cursor-pointer text-primary-600 font-medium">
          Click to browse
        </label>
        <span className="text-gray-600"> or drag and drop MP3, WAV, or M4A (max 500MB)</span>
        {file && (
          <div className="mt-4 text-left max-w-lg mx-auto space-y-2">
            <div className="text-sm text-gray-800">
              <span className="font-medium">File:</span> {file.name}
            </div>
            {durationEstimate != null && (
              <div className="text-sm text-gray-600">Estimated duration: {formatTime(durationEstimate)}</div>
            )}
            {(submitting || uploadPct > 0) && (
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
              onClick={() => void handleScan()}
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
                'Scan for Ads'
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
              <p className="mt-2 text-sm text-gray-600">{stageLabel(status)}</p>
              <div className="mt-3">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Progress</span>
                  <span>{status.progress_pct}%</span>
                </div>
                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-500 transition-all duration-500 ease-out"
                    style={{ width: `${status.progress_pct}%` }}
                  />
                </div>
              </div>
              {status.status === 'failed' && (
                <div className="mt-4">
                  <Alert type="error" message={status.error || 'Scan failed'} />
                </div>
              )}
            </>
          )}
        </div>
      )}

      {status?.status === 'complete' && totalDuration > 0 && (
        <div className="mt-8 space-y-6">
          <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-medium text-gray-900">Timeline</h2>
            <p className="text-sm text-gray-600 mt-1">
              Blue: main content. Orange: detected ads. Hover for times and confidence. Click a segment to seek the
              player.
            </p>
            <div className="mt-4 flex h-10 w-full rounded overflow-hidden border border-gray-200">
              {timeline.map((part, i) => {
                const w = ((part.end - part.start) / totalDuration) * 100;
                const title =
                  part.kind === 'ad'
                    ? `${formatTime(part.start)} – ${formatTime(part.end)} · ${part.label} · ${Math.round(
                        part.confidence * 100
                      )}%`
                    : `${formatTime(part.start)} – ${formatTime(part.end)} · Episode`;
                return (
                  <button
                    key={`${part.kind}-${i}`}
                    type="button"
                    title={title}
                    className={`h-full min-w-[2px] focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                      part.kind === 'ad' ? 'bg-orange-500 hover:bg-orange-600' : 'bg-blue-300 hover:bg-blue-400'
                    }`}
                    style={{ width: `${w}%` }}
                    onClick={() => seekTo(part.start)}
                  />
                );
              })}
            </div>
            {audioUrl && <audio ref={audioRef} controls className="mt-4 w-full max-w-xl" src={audioUrl} />}
          </div>

          <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-medium text-gray-900">Export</h2>
            <div className="mt-4 flex flex-wrap gap-4">
              <div className="flex flex-col gap-2">
                <button
                  type="button"
                  disabled={exportingClean || !jobId}
                  onClick={() => void runExport('clean')}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-md bg-gray-900 text-white text-sm font-medium hover:bg-gray-800 disabled:opacity-50"
                >
                  {exportingClean ? (
                    <span className="mr-2 inline-flex">
                      <LoadingSpinner size="sm" />
                    </span>
                  ) : null}
                  Download Clean Podcast
                </button>
                {exportClean && (
                  <span className="text-xs text-gray-600">
                    {formatTime(exportClean.duration_seconds)} · {formatBytes(exportClean.file_size_bytes)}
                  </span>
                )}
              </div>
              <div className="flex flex-col gap-2">
                <button
                  type="button"
                  disabled={exportingAds || !jobId || !(status.ad_segments && status.ad_segments.length)}
                  onClick={() => void runExport('ads_only')}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-md border border-orange-500 text-orange-700 text-sm font-medium hover:bg-orange-50 disabled:opacity-50"
                >
                  {exportingAds ? (
                    <span className="mr-2 inline-flex">
                      <LoadingSpinner size="sm" />
                    </span>
                  ) : null}
                  Download Ads Only
                </button>
                {exportAds && (
                  <span className="text-xs text-gray-600">
                    {formatTime(exportAds.duration_seconds)} · {formatBytes(exportAds.file_size_bytes)}
                  </span>
                )}
              </div>
            </div>
            {status.ad_segments && status.ad_segments.length === 0 && (
              <p className="mt-3 text-sm text-gray-500">No ad segments were detected; ads-only export is disabled.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
