import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { RecordingUpload } from '../components/transcripts/RecordingUpload';
import { ProcessingStatus } from '../components/transcripts/ProcessingStatus';
import { SpeakerLabeler } from '../components/transcripts/SpeakerLabeler';
import { TranscriptViewer } from '../components/transcripts/TranscriptViewer';
import { SummaryView } from '../components/transcripts/SummaryView';
import { ExportControls } from '../components/transcripts/ExportControls';
import type { RecordingType, TranscriptItem, TranscriptStatusResponse } from '../types/api';
import { mergeTranscriptSegments } from '../utils/transcript';

export function TranscriptsPage() {
  const navigate = useNavigate();
  const {
    uploadTranscript,
    getTranscriptStatus,
    getTranscript,
    updateTranscriptSpeakers,
    downloadTranscriptReport,
    listTranscripts,
    loading,
    error,
  } = useApi();
  const [activeTab, setActiveTab] = useState<'new' | 'resolved'>('new');
  const [transcriptId, setTranscriptId] = useState<string | null>(null);
  const [status, setStatus] = useState<TranscriptStatusResponse | null>(null);
  const [item, setItem] = useState<TranscriptItem | null>(null);
  const [resolvedItems, setResolvedItems] = useState<TranscriptItem[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const refreshResolved = async () => {
    const result = await listTranscripts({ status: 'complete', limit: 50, offset: 0 });
    if (result) {
      setResolvedItems(result.transcripts || []);
    }
  };

  useEffect(() => {
    if (activeTab !== 'resolved') return;
    refreshResolved();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  useEffect(() => {
    if (!transcriptId) return;
    let stop = false;
    let timer: number | null = null;
    const processingStatuses = new Set([
      'uploading',
      'queued',
      'transcribing',
      'diarizing',
      'matching',
      'analyzing',
    ]);

    const poll = async () => {
      const s = await getTranscriptStatus(transcriptId);
      if (stop || !s) return;
      setStatus(s);

      // Terminal states: fetch once then stop polling.
      if (!processingStatuses.has(s.status)) {
        const full = await getTranscript(transcriptId);
        if (!stop && full) setItem(full);
        if (timer !== null) {
          window.clearInterval(timer);
          timer = null;
        }
        return;
      }
    };

    poll();
    timer = window.setInterval(poll, 3000);
    return () => {
      stop = true;
      if (timer !== null) {
        window.clearInterval(timer);
      }
    };
  }, [transcriptId, getTranscriptStatus, getTranscript]);

  const startUpload = async (payload: {
    file: File;
    title?: string;
    recordingType: RecordingType;
    language: string;
  }) => {
    setSuccessMessage(null);
    setItem(null);
    setStatus(null);
    const response = await uploadTranscript({
      audioFile: payload.file,
      title: payload.title,
      recordingType: payload.recordingType,
      language: payload.language,
    });
    if (response) {
      setTranscriptId(response.transcript_id);
      setSuccessMessage(response.message);
      setActiveTab('new');
    }
  };

  const handleProceed = async (labels: { id: string; label: string }[]) => {
    if (!transcriptId) return;
    const result = await updateTranscriptSpeakers(transcriptId, {
      speakers: labels,
      proceed_to_analysis: true,
    });
    if (result) {
      setSuccessMessage(result.message);
      const full = await getTranscript(transcriptId);
      if (full) setItem(full);
    }
  };

  const transcriptTextForCopy = useMemo(() => {
    if (!item?.transcript) return '';
    const merged = mergeTranscriptSegments(item.transcript);
    return merged.map((x) => `${x.speaker_id} [${Math.round(x.start_ms / 1000)}s] ${x.text}`).join('\n');
  }, [item]);

  const downloadReport = async (format: 'pdf' | 'json') => {
    if (!transcriptId) return;
    const blob = await downloadTranscriptReport(transcriptId, format);
    if (!blob) return;
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${transcriptId}.${format}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const copyTranscript = () => {
    if (!transcriptTextForCopy) return;
    navigator.clipboard.writeText(transcriptTextForCopy);
    setSuccessMessage('Transcript copied to clipboard.');
  };

  const generateFromTranscript = () => {
    if (!transcriptId || !item || item.speakers.length === 0) return;
    navigate(`/generate?from=${encodeURIComponent(transcriptId)}`);
  };

  const openResolvedTranscript = async (selectedId: string) => {
    setTranscriptId(selectedId);
    const s = await getTranscriptStatus(selectedId);
    if (s) setStatus(s);
    const full = await getTranscript(selectedId);
    if (full) {
      setItem(full);
      setActiveTab('new');
      setSuccessMessage(`Loaded transcript: ${full.title}`);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Transcripts</h1>
        <p className="mt-2 text-gray-600">
          Upload meetings, calls, voice memos, and interviews for transcription and analysis.
        </p>
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setActiveTab('new')}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            activeTab === 'new'
              ? 'bg-primary-600 text-white'
              : 'bg-white text-gray-700 border border-gray-300'
          }`}
        >
          New Transcript
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('resolved')}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            activeTab === 'resolved'
              ? 'bg-primary-600 text-white'
              : 'bg-white text-gray-700 border border-gray-300'
          }`}
        >
          Resolved History
        </button>
      </div>

      {error && <Alert type="error" message={error} />}
      {successMessage && (
        <Alert type="success" message={successMessage} onClose={() => setSuccessMessage(null)} />
      )}

      {activeTab === 'new' && (
        <>
          <RecordingUpload isLoading={loading && !transcriptId} onSubmit={startUpload} />

          {status && status.status !== 'complete' && status.status !== 'awaiting_labels' && (
            <ProcessingStatus status={status} />
          )}

          {status?.status === 'awaiting_labels' && item?.speakers && (
            <SpeakerLabeler speakers={item.speakers} onProceed={handleProceed} isLoading={loading} />
          )}

          {loading && transcriptId && !item ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="lg" />
            </div>
          ) : null}

          {item && (
            <>
              <SummaryView analysis={item.analysis} recordingType={item.recording_type} />
              <TranscriptViewer transcript={item.transcript || []} />
              <ExportControls
                onDownloadPdf={() => downloadReport('pdf')}
                onDownloadJson={() => downloadReport('json')}
                onCopyTranscript={copyTranscript}
                onGenerateFromTranscript={generateFromTranscript}
                canGenerateFromTranscript={item.speakers.length > 0}
                isLoading={loading}
              />
            </>
          )}
        </>
      )}

      {activeTab === 'resolved' && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Resolved Transcripts</h2>
            <button
              type="button"
              onClick={refreshResolved}
              className="px-3 py-2 text-sm border rounded-md bg-gray-50 hover:bg-gray-100"
            >
              Refresh
            </button>
          </div>

          {resolvedItems.length === 0 ? (
            <p className="text-sm text-gray-500">No completed transcripts yet.</p>
          ) : (
            <div className="space-y-2">
              {resolvedItems.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => openResolvedTranscript(t.id)}
                  className="w-full text-left border rounded-md p-3 hover:bg-gray-50"
                >
                  <div className="font-medium text-gray-900">{t.title || t.file_name}</div>
                  <div className="text-sm text-gray-600">
                    {t.recording_type} • {t.language} • {new Date(t.created_at).toLocaleString()}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

