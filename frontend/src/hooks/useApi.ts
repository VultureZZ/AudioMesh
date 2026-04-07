/**
 * Custom hook that wraps API client with loading/error states
 */

import { useState, useCallback } from 'react';
import { apiClient } from '../services/api';
import {
  SpeechGenerateRequest,
  SpeechGenerateResponse,
  VoiceListResponse,
  VoiceCreateResponse,
  HealthCheckResponse,
  PodcastScriptRequest,
  PodcastScriptResponse,
  PodcastGenerateRequest,
  PodcastGenerateResponse,
  PodcastListResponse,
  MusicCoverGenerateParams,
  MusicGenerateRequest,
  MusicGenerateResponse,
  MusicHealthResponse,
  MusicLyricsRequest,
  MusicLyricsResponse,
  MusicSimpleRequest,
  MusicStatusResponse,
  MusicPreset,
  MusicPresetListResponse,
  MusicPresetRequest,
  MusicHistoryItem,
  MusicHistoryListResponse,
  VoiceProfileResponse,
  VoiceProfileRequest,
  VoiceProfileApplyRequest,
  VoiceProfileFromAudioResponse,
  VoiceUpdateRequest,
  VoiceUpdateResponse,
  AudioClipRange,
  RecordingType,
  TranscriptUploadResponse,
  TranscriptStatusResponse,
  TranscriptItem,
  TranscriptListResponse,
} from '../types/api';

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async <T,>(fn: () => Promise<T>): Promise<T | null> => {
      setLoading(true);
      setError(null);
      try {
        const result = await fn();
        return result;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
        setError(errorMessage);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const healthCheck = useCallback(async (): Promise<HealthCheckResponse | null> => {
    return execute(() => apiClient.healthCheck());
  }, [execute]);

  const generateSpeech = useCallback(
    async (request: SpeechGenerateRequest): Promise<SpeechGenerateResponse | null> => {
      return execute(() => apiClient.generateSpeech(request));
    },
    [execute]
  );

  const generateSpeechWithProgress = useCallback(
    async (
      request: SpeechGenerateRequest,
      onProgress: (current: number, total: number, message: string) => void
    ): Promise<SpeechGenerateResponse | null> => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiClient.generateSpeechWithProgress(request, onProgress);
        return result;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
        setError(errorMessage);
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const downloadAudio = useCallback(
    async (filename: string): Promise<Blob | null> => {
      return execute(() => apiClient.downloadAudio(filename));
    },
    [execute]
  );

  const listVoices = useCallback(async (): Promise<VoiceListResponse | null> => {
    return execute(() => apiClient.listVoices());
  }, [execute]);

  const createVoice = useCallback(
    async (
      name: string,
      description: string | undefined,
      files: File[],
      keywords?: string,
      languageCode?: string,
      gender?: string,
      image?: File
    ): Promise<VoiceCreateResponse | null> => {
      return execute(() =>
        apiClient.createVoice({
          name,
          description,
          creation_source: 'audio',
          audio_files: files,
          keywords,
          language_code: languageCode,
          gender,
          image,
        })
      );
    },
    [execute]
  );

  const createVoiceUnified = useCallback(
    async (
      params: Parameters<typeof apiClient.createVoice>[0]
    ): Promise<VoiceCreateResponse | null> => {
      return execute(() => apiClient.createVoice(params));
    },
    [execute]
  );

  const createVoiceFromClips = useCallback(
    async (
      name: string,
      description: string | undefined,
      audioFile: File,
      clipRanges: AudioClipRange[],
      keywords?: string,
      languageCode?: string,
      gender?: string
    ): Promise<VoiceCreateResponse | null> => {
      return execute(() =>
        apiClient.createVoiceFromClips(name, description, audioFile, clipRanges, keywords, languageCode, gender)
      );
    },
    [execute]
  );

  const deleteVoice = useCallback(
    async (voiceId: string): Promise<boolean> => {
      const result = await execute(() => apiClient.deleteVoice(voiceId));
      return result !== null;
    },
    [execute]
  );

  const generatePodcastScript = useCallback(
    async (request: PodcastScriptRequest): Promise<PodcastScriptResponse | null> => {
      return execute(() => apiClient.generatePodcastScript(request));
    },
    [execute]
  );

  const generatePodcastAudio = useCallback(
    async (request: PodcastGenerateRequest): Promise<PodcastGenerateResponse | null> => {
      return execute(() => apiClient.generatePodcastAudio(request));
    },
    [execute]
  );

  const generateMusic = useCallback(
    async (request: MusicGenerateRequest): Promise<MusicGenerateResponse | null> => {
      return execute(() => apiClient.generateMusic(request));
    },
    [execute]
  );

  const generateMusicCover = useCallback(
    async (referenceAudio: File, params: MusicCoverGenerateParams): Promise<MusicGenerateResponse | null> => {
      return execute(() => apiClient.generateMusicCover(referenceAudio, params));
    },
    [execute]
  );

  const simpleGenerateMusic = useCallback(
    async (request: MusicSimpleRequest): Promise<MusicGenerateResponse | null> => {
      return execute(() => apiClient.simpleGenerateMusic(request));
    },
    [execute]
  );

  const getMusicStatus = useCallback(
    async (taskId: string): Promise<MusicStatusResponse | null> => {
      return execute(() => apiClient.getMusicStatus(taskId));
    },
    [execute]
  );

  const generateLyrics = useCallback(
    async (request: MusicLyricsRequest): Promise<MusicLyricsResponse | null> => {
      return execute(() => apiClient.generateLyrics(request));
    },
    [execute]
  );

  const downloadMusic = useCallback(
    async (filename: string): Promise<Blob | null> => {
      return execute(() => apiClient.downloadMusic(filename));
    },
    [execute]
  );

  const checkMusicHealth = useCallback(async (): Promise<MusicHealthResponse | null> => {
    return execute(() => apiClient.checkMusicHealth());
  }, [execute]);

  const listMusicPresets = useCallback(async (): Promise<MusicPresetListResponse | null> => {
    return execute(() => apiClient.listMusicPresets());
  }, [execute]);

  const createMusicPreset = useCallback(
    async (request: MusicPresetRequest): Promise<MusicPreset | null> => {
      return execute(() => apiClient.createMusicPreset(request));
    },
    [execute]
  );

  const updateMusicPreset = useCallback(
    async (presetId: string, request: MusicPresetRequest): Promise<MusicPreset | null> => {
      return execute(() => apiClient.updateMusicPreset(presetId, request));
    },
    [execute]
  );

  const deleteMusicPreset = useCallback(
    async (presetId: string): Promise<boolean> => {
      const result = await execute(() => apiClient.deleteMusicPreset(presetId));
      return result !== null;
    },
    [execute]
  );

  const listMusicHistory = useCallback(
    async (limit = 50): Promise<MusicHistoryListResponse | null> => {
      return execute(() => apiClient.listMusicHistory(limit));
    },
    [execute]
  );

  const getMusicHistoryItem = useCallback(
    async (historyId: string): Promise<MusicHistoryItem | null> => {
      return execute(() => apiClient.getMusicHistoryItem(historyId));
    },
    [execute]
  );

  const deleteMusicHistoryItem = useCallback(
    async (historyId: string): Promise<boolean> => {
      const result = await execute(() => apiClient.deleteMusicHistoryItem(historyId));
      return result !== null;
    },
    [execute]
  );

  const downloadPodcastAudio = useCallback(
    async (filename: string): Promise<Blob | null> => {
      return execute(() => apiClient.downloadPodcastAudio(filename));
    },
    [execute]
  );

  const listPodcasts = useCallback(
    async (query?: string): Promise<PodcastListResponse | null> => {
      return execute(() => apiClient.listPodcasts(query));
    },
    [execute]
  );

  const deletePodcast = useCallback(
    async (podcastId: string): Promise<boolean> => {
      const result = await execute(() => apiClient.deletePodcast(podcastId));
      return result !== null;
    },
    [execute]
  );

  const downloadPodcastById = useCallback(
    async (podcastId: string): Promise<Blob | null> => {
      return execute(() => apiClient.downloadPodcastById(podcastId));
    },
    [execute]
  );

  const analyzeVoiceProfileFromAudio = useCallback(
    async (
      audioFile: File,
      keywords?: string,
      ollamaUrl?: string,
      ollamaModel?: string
    ): Promise<VoiceProfileFromAudioResponse | null> => {
      return execute(() => apiClient.analyzeVoiceProfileFromAudio(audioFile, keywords, ollamaUrl, ollamaModel));
    },
    [execute]
  );

  const applyVoiceProfile = useCallback(
    async (voiceId: string, request: VoiceProfileApplyRequest): Promise<VoiceProfileResponse | null> => {
      return execute(() => apiClient.applyVoiceProfile(voiceId, request));
    },
    [execute]
  );

  const uploadTranscript = useCallback(
    async (params: {
      audioFile: File;
      title?: string;
      language?: string;
      recordingType?: RecordingType;
    }): Promise<TranscriptUploadResponse | null> => {
      return execute(() => apiClient.uploadTranscript(params));
    },
    [execute]
  );

  const getTranscriptStatus = useCallback(
    async (transcriptId: string): Promise<TranscriptStatusResponse | null> => {
      return execute(() => apiClient.getTranscriptStatus(transcriptId));
    },
    [execute]
  );

  const getTranscript = useCallback(
    async (transcriptId: string): Promise<TranscriptItem | null> => {
      return execute(() => apiClient.getTranscript(transcriptId));
    },
    [execute]
  );

  const updateTranscriptSpeakers = useCallback(
    async (
      transcriptId: string,
      payload: { speakers: { id: string; label: string }[]; proceed_to_analysis: boolean }
    ): Promise<{ transcript_id: string; status: string; message: string } | null> => {
      return execute(() => apiClient.updateTranscriptSpeakers(transcriptId, payload));
    },
    [execute]
  );

  const listTranscripts = useCallback(
    async (params?: {
      limit?: number;
      offset?: number;
      status?: string;
      recording_type?: RecordingType;
    }): Promise<TranscriptListResponse | null> => {
      return execute(() => apiClient.listTranscripts(params));
    },
    [execute]
  );

  const deleteTranscript = useCallback(
    async (transcriptId: string): Promise<boolean> => {
      const result = await execute(() => apiClient.deleteTranscript(transcriptId));
      return result !== null;
    },
    [execute]
  );

  const downloadTranscriptReport = useCallback(
    async (transcriptId: string, format: 'pdf' | 'json' | 'markdown'): Promise<Blob | null> => {
      return execute(() => apiClient.downloadTranscriptReport(transcriptId, format));
    },
    [execute]
  );

  const updateVoice = useCallback(
    async (voiceId: string, request: VoiceUpdateRequest): Promise<VoiceUpdateResponse | null> => {
      return execute(() => apiClient.updateVoice(voiceId, request));
    },
    [execute]
  );

  const uploadVoiceImage = useCallback(
    async (voiceId: string, file: File): Promise<VoiceUpdateResponse | null> => {
      return execute(() => apiClient.uploadVoiceImage(voiceId, file));
    },
    [execute]
  );

  const getVoiceProfile = useCallback(
    async (voiceId: string): Promise<VoiceProfileResponse | null> => {
      return execute(() => apiClient.getVoiceProfile(voiceId));
    },
    [execute]
  );

  const createOrUpdateVoiceProfile = useCallback(
    async (voiceId: string, request: VoiceProfileRequest): Promise<VoiceProfileResponse | null> => {
      return execute(() => apiClient.createOrUpdateVoiceProfile(voiceId, request));
    },
    [execute]
  );

  const updateVoiceProfileKeywords = useCallback(
    async (voiceId: string, request: VoiceProfileRequest): Promise<VoiceProfileResponse | null> => {
      return execute(() => apiClient.updateVoiceProfileKeywords(voiceId, request));
    },
    [execute]
  );

  const generateVoiceProfile = useCallback(
    async (voiceId: string, request: VoiceProfileRequest): Promise<VoiceProfileResponse | null> => {
      return execute(() => apiClient.generateVoiceProfile(voiceId, request));
    },
    [execute]
  );

  return {
    loading,
    error,
    healthCheck,
    generateSpeech,
    generateSpeechWithProgress,
    downloadAudio,
    listVoices,
    createVoice,
    createVoiceUnified,
    createVoiceFromClips,
    deleteVoice,
    updateVoice,
    uploadVoiceImage,
    getVoiceProfile,
    createOrUpdateVoiceProfile,
    updateVoiceProfileKeywords,
    generateVoiceProfile,
    generatePodcastScript,
    generatePodcastAudio,
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
    getMusicHistoryItem,
    deleteMusicHistoryItem,
    downloadPodcastAudio,
    listPodcasts,
    deletePodcast,
    downloadPodcastById,
    analyzeVoiceProfileFromAudio,
    applyVoiceProfile,
    uploadTranscript,
    getTranscriptStatus,
    getTranscript,
    updateTranscriptSpeakers,
    listTranscripts,
    deleteTranscript,
    downloadTranscriptReport,
  };
}