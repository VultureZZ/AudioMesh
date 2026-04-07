/**
 * Podcast Library page - list/search/delete saved podcasts
 */

import { useEffect, useMemo, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { useSettings } from '../hooks/useSettings';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { AudioPlayer } from '../components/AudioPlayer';
import type { PodcastItem } from '../types/api';

export function PodcastsLibraryPage() {
  const { settings } = useSettings();
  const { listPodcasts, deletePodcast, downloadPodcastById, loading, error } = useApi();

  const [query, setQuery] = useState('');
  const [items, setItems] = useState<PodcastItem[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  const fetchPodcasts = async (q?: string) => {
    const resp = await listPodcasts(q);
    if (resp) {
      setItems(resp.podcasts);
    }
  };

  useEffect(() => {
    fetchPodcasts('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      fetchPodcasts(query);
    }, 250);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  const hasItems = items.length > 0;
  const headerText = useMemo(() => {
    if (!query.trim()) return 'Saved podcasts';
    return `Results for "${query.trim()}"`;
  }, [query]);

  const handleDelete = async (podcastId: string) => {
    if (!confirm('Delete this podcast? This will remove it from the library and delete its audio file.')) return;
    setDeletingId(podcastId);
    setSuccessMessage(null);
    const ok = await deletePodcast(podcastId);
    setDeletingId(null);
    if (ok) {
      setSuccessMessage('Podcast deleted');
      fetchPodcasts(query);
    }
  };

  const handleDownload = async (podcastId: string, title: string) => {
    setDownloadingId(podcastId);
    const blob = await downloadPodcastById(podcastId);
    setDownloadingId(null);
    if (!blob) return;

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title || podcastId}.wav`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Podcast Library</h1>
        <p className="mt-2 text-gray-600">Search and manage saved podcasts</p>
      </div>

      {error && <Alert type="error" message={error} />}
      {successMessage && <Alert type="success" message={successMessage} onClose={() => setSuccessMessage(null)} />}

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <Input
          label="Search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by title, URL, or voice name..."
        />

        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">{headerText}</div>
          <Button variant="secondary" onClick={() => fetchPodcasts(query)} isLoading={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {loading && !hasItems ? (
        <div className="flex justify-center items-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : hasItems ? (
        <div className="space-y-4">
          {items.map((p) => {
            const audioUrl = `${settings.apiEndpoint}${p.audio_url || `/api/v1/podcasts/${p.id}/download`}`;
            const subtitleParts = [
              p.genre ? `Genre: ${p.genre}` : null,
              p.duration ? `Duration: ${p.duration}` : null,
              p.voices?.length ? `Voices: ${p.voices.join(', ')}` : null,
            ].filter(Boolean);

            return (
              <div key={p.id} className="bg-white rounded-lg shadow p-6 space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <h2 className="text-lg font-semibold text-gray-900 truncate">{p.title || p.id}</h2>
                    {subtitleParts.length > 0 && (
                      <p className="mt-1 text-sm text-gray-600">{subtitleParts.join(' • ')}</p>
                    )}
                    {p.source_url && (
                      <p className="mt-1 text-xs text-gray-500 truncate">{p.source_url}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={() => handleDownload(p.id, p.title)}
                      isLoading={downloadingId === p.id}
                    >
                      Download
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => handleDelete(p.id)}
                      isLoading={deletingId === p.id}
                    >
                      Delete
                    </Button>
                  </div>
                </div>

                <AudioPlayer src={audioUrl} filename={p.title || p.id} />
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-500">
          <p>No saved podcasts found</p>
        </div>
      )}
    </div>
  );
}

