/**
 * API client service using Axios
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import {
  SpeechGenerateRequest,
  SpeechGenerateResponse,
  VoiceListResponse,
  VoiceCreateResponse,
  HealthCheckResponse,
  ErrorResponse,
  PodcastScriptRequest,
  PodcastScriptResponse,
  PodcastGenerateRequest,
  PodcastGenerateResponse,
  PodcastProductionRequest,
  PodcastProductionStatusResponse,
  PodcastProductionSubmitResponse,
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
  PodcastAdScanSubmitResponse,
  PodcastAdScanStatusResponse,
  PodcastAdExportResponse,
  SpeakerIsolationSubmitResponse,
  SpeakerIsolationStatusResponse,
} from '../types/api';
import { AppSettings } from '../types/settings';

class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;
  private apiKey?: string;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    this.apiKey = undefined;

    this.client = axios.create({
      baseURL: this.baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for API key and FormData handling
    this.client.interceptors.request.use((config) => {
      if (this.apiKey) {
        config.headers['X-API-Key'] = this.apiKey;
      }
      // Let browser set Content-Type with boundary for FormData; explicit multipart/form-data without boundary breaks parsing
      if (config.data instanceof FormData) {
        delete config.headers['Content-Type'];
      }
      return config;
    });

    // Response interceptor for error handling (blob bodies need text parsing — e.g. speech/voice binary routes)
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError<ErrorResponse>) => {
        if (error.response?.data instanceof Blob) {
          const text = await error.response.data.text();
          let message = error.message;
          try {
            const j = JSON.parse(text) as { detail?: string; error?: string };
            message = j.detail || j.error || message;
          } catch {
            if (text.trimStart().startsWith('<!') || text.trimStart().toLowerCase().startsWith('<html')) {
              message = 'Server returned HTML instead of an API response (check API base URL and proxy).';
            } else if (text.length > 0) {
              message = text.slice(0, 280);
            }
          }
          throw new Error(message);
        }
        if (error.response) {
          const errorData = error.response.data;
          const errorMessage = errorData?.detail || errorData?.error || error.message;
          throw new Error(typeof errorMessage === 'string' ? errorMessage : error.message);
        } else if (error.request) {
          throw new Error('Network error: Could not reach the API server');
        } else {
          throw new Error(error.message || 'An unexpected error occurred');
        }
      }
    );
  }

  /**
   * Update API configuration
   */
  updateConfig(settings: AppSettings): void {
    this.baseURL = settings.apiEndpoint;
    this.apiKey = settings.apiKey;
    this.client.defaults.baseURL = this.baseURL;
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>('/health');
    return response.data;
  }

  /**
   * Generate speech from text
   */
  async generateSpeech(request: SpeechGenerateRequest): Promise<SpeechGenerateResponse> {
    const response = await this.client.post<SpeechGenerateResponse>(
      '/api/v1/speech/generate',
      request
    );
    return response.data;
  }

  /**
   * Generate speech with progress updates via SSE.
   * Calls onProgress(current, total, message) for each progress event.
   */
  async generateSpeechWithProgress(
    request: SpeechGenerateRequest,
    onProgress: (current: number, total: number, message: string) => void
  ): Promise<SpeechGenerateResponse | null> {
    const url = `${this.baseURL}/api/v1/speech/generate-stream`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || err.error || 'Speech generation failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          if (jsonStr === '[DONE]' || jsonStr.trim() === '') continue;
          try {
            const data = JSON.parse(jsonStr) as {
              type: string;
              current?: number;
              total?: number;
              message?: string;
              success?: boolean;
              audio_url?: string;
              filename?: string;
              detail?: string;
            };
            if (data.type === 'progress' && data.current != null && data.total != null) {
              onProgress(data.current, data.total, data.message ?? '');
            } else if (data.type === 'complete' && data.success && data.audio_url) {
              return {
                success: true,
                message: 'Speech generated successfully',
                audio_url: data.audio_url,
              };
            } else if (data.type === 'error' && data.detail) {
              throw new Error(data.detail);
            }
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      }
    }

    throw new Error('Generation completed without result');
  }

  /**
   * Download generated audio file
   */
  async downloadAudio(filename: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/speech/download/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * List all available voices
   */
  async listVoices(): Promise<VoiceListResponse> {
    const response = await this.client.get<VoiceListResponse>('/api/v1/voices');
    return response.data;
  }

  /**
   * Short cached TTS preview for a voice (WAV). Server stores under custom_voices/voice_samples/.
   * Rejects if the server returns JSON/HTML instead of audio (common misconfiguration when the blob would not play).
   */
  async getVoiceSample(voiceId: string, language?: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/voices/${encodeURIComponent(voiceId)}/sample`, {
      responseType: 'blob',
      params: language ? { language } : undefined,
    });
    const blob = response.data as Blob;
    const ct = (response.headers['content-type'] || '').toLowerCase();

    if (ct.includes('application/json') || blob.type === 'application/json') {
      const text = await blob.text();
      let j: { detail?: string; error?: string };
      try {
        j = JSON.parse(text) as { detail?: string; error?: string };
      } catch {
        throw new Error(text.slice(0, 200) || 'Voice sample response was not JSON');
      }
      throw new Error(j.detail || j.error || 'Voice sample request failed');
    }

    if (blob.size < 32) {
      throw new Error('Voice sample response was empty or too small (check API and voice id).');
    }

    const head = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
    const looksLikeWav = head[0] === 0x52 && head[1] === 0x49 && head[2] === 0x46 && head[3] === 0x46;
    const snippet = await blob.slice(0, 64).text();
    if (!looksLikeWav && (snippet.trimStart().startsWith('<!') || snippet.trimStart().toLowerCase().startsWith('<html'))) {
      throw new Error(
        'Server returned a web page instead of audio. Set Settings API URL to your AudioMesh backend (not the Vite port).'
      );
    }

    const mime =
      ct.includes('audio/') || ct.includes('octet-stream')
        ? ct.split(';')[0].trim()
        : 'audio/wav';
    return blob.type && blob.type.startsWith('audio/') ? blob : new Blob([blob], { type: mime });
  }

  /**
   * Create a voice: from audio files, from clips, or from a text description (VoiceDesign).
   */
  async createVoice(params: {
    name: string;
    description?: string;
    creation_source: 'audio' | 'clips' | 'prompt';
    audio_files?: File[];
    audio_file?: File;
    clip_ranges?: AudioClipRange[];
    voice_design_prompt?: string;
    keywords?: string;
    language_code?: string;
    gender?: string;
    image?: File;
  }): Promise<VoiceCreateResponse> {
    const formData = new FormData();
    formData.append('name', params.name);
    if (params.description) {
      formData.append('description', params.description);
    }
    formData.append('creation_source', params.creation_source);
    if (params.keywords) {
      formData.append('keywords', params.keywords);
    }
    if (params.language_code) {
      formData.append('language_code', params.language_code);
    }
    if (params.gender) {
      formData.append('gender', params.gender);
    }
    if (params.image) {
      formData.append('image', params.image);
    }
    if (params.creation_source === 'audio' && params.audio_files?.length) {
      params.audio_files.forEach((file) => formData.append('audio_files', file));
    }
    if (params.creation_source === 'clips') {
      if (params.audio_file) {
        formData.append('audio_file', params.audio_file);
      }
      if (params.clip_ranges?.length) {
        formData.append('clip_ranges', JSON.stringify(params.clip_ranges));
      }
    }
    if (params.creation_source === 'prompt' && params.voice_design_prompt) {
      formData.append('voice_design_prompt', params.voice_design_prompt);
    }

    const response = await this.client.post<VoiceCreateResponse>('/api/v1/voices', formData);
    return response.data;
  }

  /**
   * Upload or replace avatar image for a custom voice
   */
  async uploadVoiceImage(voiceId: string, file: File): Promise<VoiceUpdateResponse> {
    const formData = new FormData();
    formData.append('image', file);
    const response = await this.client.put<VoiceUpdateResponse>(
      `/api/v1/voices/${voiceId}/image`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  }

  /**
   * Create a voice from clips (convenience wrapper for createVoice with creation_source=clips).
   */
  async createVoiceFromClips(
    name: string,
    description: string | undefined,
    audioFile: File,
    clipRanges: AudioClipRange[],
    keywords?: string,
    languageCode?: string,
    gender?: string,
    image?: File
  ): Promise<VoiceCreateResponse> {
    return this.createVoice({
      name,
      description,
      creation_source: 'clips',
      audio_file: audioFile,
      clip_ranges: clipRanges,
      keywords,
      language_code: languageCode,
      gender,
      image,
    });
  }

  /**
   * Delete a custom voice
   */
  async deleteVoice(voiceId: string): Promise<void> {
    await this.client.delete(`/api/v1/voices/${voiceId}`);
  }

  /**
   * Update voice details (name and/or description)
   */
  async updateVoice(
    voiceId: string,
    request: VoiceUpdateRequest
  ): Promise<VoiceUpdateResponse> {
    const response = await this.client.put<VoiceUpdateResponse>(
      `/api/v1/voices/${voiceId}`,
      request
    );
    return response.data;
  }

  /**
   * Get voice profile
   */
  async getVoiceProfile(voiceId: string): Promise<VoiceProfileResponse> {
    const response = await this.client.get<VoiceProfileResponse>(
      `/api/v1/voices/${voiceId}/profile`
    );
    return response.data;
  }

  /**
   * Create or update voice profile
   */
  async createOrUpdateVoiceProfile(
    voiceId: string,
    request: VoiceProfileRequest
  ): Promise<VoiceProfileResponse> {
    const response = await this.client.post<VoiceProfileResponse>(
      `/api/v1/voices/${voiceId}/profile`,
      request
    );
    return response.data;
  }

  /**
   * Update voice profile keywords
   */
  async updateVoiceProfileKeywords(
    voiceId: string,
    request: VoiceProfileRequest
  ): Promise<VoiceProfileResponse> {
    const response = await this.client.put<VoiceProfileResponse>(
      `/api/v1/voices/${voiceId}/profile/keywords`,
      request
    );
    return response.data;
  }

  /**
   * Manually generate voice profile
   */
  async generateVoiceProfile(
    voiceId: string,
    request: VoiceProfileRequest
  ): Promise<VoiceProfileResponse> {
    const response = await this.client.post<VoiceProfileResponse>(
      `/api/v1/voices/${voiceId}/profile/generate`,
      request
    );
    return response.data;
  }

  /**
   * Generate podcast script from article URL
   */
  async generatePodcastScript(request: PodcastScriptRequest): Promise<PodcastScriptResponse> {
    const response = await this.client.post<PodcastScriptResponse>(
      '/api/v1/podcast/generate-script',
      request
    );
    return response.data;
  }

  /**
   * Generate podcast audio from script
   */
  async generatePodcastAudio(request: PodcastGenerateRequest): Promise<PodcastGenerateResponse> {
    const response = await this.client.post<PodcastGenerateResponse>(
      '/api/v1/podcast/generate',
      request
    );
    return response.data;
  }

  /**
   * Submit production-mode podcast generation task
   */
  async generatePodcastProduction(
    request: PodcastProductionRequest
  ): Promise<PodcastProductionSubmitResponse> {
    const response = await this.client.post<PodcastProductionSubmitResponse>(
      '/api/v1/podcast/generate-production',
      request
    );
    return response.data;
  }

  /**
   * Poll production-mode podcast task status
   */
  async getPodcastProductionStatus(taskId: string): Promise<PodcastProductionStatusResponse> {
    const response = await this.client.get<PodcastProductionStatusResponse>(
      `/api/v1/podcast/status/${taskId}`
    );
    return response.data;
  }

  /**
   * Submit custom ACE-Step music generation task
   */
  async generateMusic(request: MusicGenerateRequest): Promise<MusicGenerateResponse> {
    const response = await this.client.post<MusicGenerateResponse>('/api/v1/music/generate', request);
    return response.data;
  }

  /**
   * Submit ACE-Step cover-mode generation task with reference audio
   */
  async generateMusicCover(referenceAudio: File, params: MusicCoverGenerateParams): Promise<MusicGenerateResponse> {
    const formData = new FormData();
    formData.append('reference_audio', referenceAudio);
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      formData.append(key, String(value));
    });

    const response = await this.client.post<MusicGenerateResponse>('/api/v1/music/cover-generate', formData);
    return response.data;
  }

  /**
   * Submit simple description-driven music generation task
   */
  async simpleGenerateMusic(request: MusicSimpleRequest): Promise<MusicGenerateResponse> {
    const response = await this.client.post<MusicGenerateResponse>('/api/v1/music/simple-generate', request);
    return response.data;
  }

  /**
   * Poll music generation status
   */
  async getMusicStatus(taskId: string): Promise<MusicStatusResponse> {
    const response = await this.client.get<MusicStatusResponse>(`/api/v1/music/status/${taskId}`);
    return response.data;
  }

  /**
   * Generate lyrics with LLM assistance
   */
  async generateLyrics(request: MusicLyricsRequest): Promise<MusicLyricsResponse> {
    const response = await this.client.post<MusicLyricsResponse>('/api/v1/music/generate-lyrics', request);
    return response.data;
  }

  /**
   * Download generated music file
   */
  async downloadMusic(filename: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/music/download/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Check ACE-Step music service health
   */
  async checkMusicHealth(): Promise<MusicHealthResponse> {
    const response = await this.client.get<MusicHealthResponse>('/api/v1/music/health');
    return response.data;
  }

  /**
   * List saved music presets
   */
  async listMusicPresets(): Promise<MusicPresetListResponse> {
    const response = await this.client.get<MusicPresetListResponse>('/api/v1/music/presets');
    return response.data;
  }

  /**
   * Create a music preset
   */
  async createMusicPreset(request: MusicPresetRequest): Promise<MusicPreset> {
    const response = await this.client.post<MusicPreset>('/api/v1/music/presets', request);
    return response.data;
  }

  /**
   * Update a music preset
   */
  async updateMusicPreset(presetId: string, request: MusicPresetRequest): Promise<MusicPreset> {
    const response = await this.client.put<MusicPreset>(`/api/v1/music/presets/${presetId}`, request);
    return response.data;
  }

  /**
   * Delete a music preset
   */
  async deleteMusicPreset(presetId: string): Promise<void> {
    await this.client.delete(`/api/v1/music/presets/${presetId}`);
  }

  /**
   * List music generation history
   */
  async listMusicHistory(limit = 50): Promise<MusicHistoryListResponse> {
    const response = await this.client.get<MusicHistoryListResponse>('/api/v1/music/history', {
      params: { limit },
    });
    return response.data;
  }

  /**
   * Fetch one history item
   */
  async getMusicHistoryItem(historyId: string): Promise<MusicHistoryItem> {
    const response = await this.client.get<MusicHistoryItem>(`/api/v1/music/history/${historyId}`);
    return response.data;
  }

  /**
   * Delete one history item
   */
  async deleteMusicHistoryItem(historyId: string): Promise<void> {
    await this.client.delete(`/api/v1/music/history/${historyId}`);
  }

  /**
   * Download generated podcast audio file
   */
  async downloadPodcastAudio(filename: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/podcast/download/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * List/search saved podcasts
   */
  async listPodcasts(query?: string): Promise<PodcastListResponse> {
    const response = await this.client.get<PodcastListResponse>('/api/v1/podcasts', {
      params: query ? { query } : undefined,
    });
    return response.data;
  }

  /**
   * Delete a saved podcast
   */
  async deletePodcast(podcastId: string): Promise<void> {
    await this.client.delete(`/api/v1/podcasts/${podcastId}`);
  }

  /**
   * Download saved podcast audio by id
   */
  async downloadPodcastById(podcastId: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/podcasts/${podcastId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Analyze audio file to derive a voice profile
   */
  async analyzeVoiceProfileFromAudio(
    audioFile: File,
    keywords?: string,
    ollamaUrl?: string,
    ollamaModel?: string
  ): Promise<VoiceProfileFromAudioResponse> {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    if (keywords) formData.append('keywords', keywords);
    if (ollamaUrl) formData.append('ollama_url', ollamaUrl);
    if (ollamaModel) formData.append('ollama_model', ollamaModel);

    const response = await this.client.post<VoiceProfileFromAudioResponse>(
      '/api/v1/voices/profile/analyze-audio',
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  }

  /**
   * Apply a full voice profile payload to a voice
   */
  async applyVoiceProfile(voiceId: string, request: VoiceProfileApplyRequest): Promise<VoiceProfileResponse> {
    const response = await this.client.put<VoiceProfileResponse>(`/api/v1/voices/${voiceId}/profile`, request);
    return response.data;
  }

  async uploadTranscript(params: {
    audioFile: File;
    title?: string;
    language?: string;
    recordingType?: RecordingType;
  }): Promise<TranscriptUploadResponse> {
    const formData = new FormData();
    formData.append('audio_file', params.audioFile);
    if (params.title) formData.append('title', params.title);
    if (params.language) formData.append('language', params.language);
    if (params.recordingType) formData.append('recording_type', params.recordingType);
    const response = await this.client.post<TranscriptUploadResponse>(
      '/api/v1/transcripts/upload',
      formData
    );
    return response.data;
  }

  async getTranscriptStatus(transcriptId: string): Promise<TranscriptStatusResponse> {
    const response = await this.client.get<TranscriptStatusResponse>(
      `/api/v1/transcripts/${transcriptId}/status`
    );
    return response.data;
  }

  async getTranscript(transcriptId: string): Promise<TranscriptItem> {
    const response = await this.client.get<TranscriptItem>(`/api/v1/transcripts/${transcriptId}`);
    return response.data;
  }

  async updateTranscriptSpeakers(
    transcriptId: string,
    payload: { speakers: { id: string; label: string }[]; proceed_to_analysis: boolean }
  ): Promise<{ transcript_id: string; status: string; message: string }> {
    const response = await this.client.patch<{ transcript_id: string; status: string; message: string }>(
      `/api/v1/transcripts/${transcriptId}/speakers`,
      payload
    );
    return response.data;
  }

  async listTranscripts(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    recording_type?: RecordingType;
  }): Promise<TranscriptListResponse> {
    const response = await this.client.get<TranscriptListResponse>('/api/v1/transcripts', { params });
    return response.data;
  }

  async deleteTranscript(transcriptId: string): Promise<void> {
    await this.client.delete(`/api/v1/transcripts/${transcriptId}`);
  }

  async downloadTranscriptReport(transcriptId: string, format: 'pdf' | 'json' | 'markdown'): Promise<Blob> {
    const response = await this.client.get(`/api/v1/transcripts/${transcriptId}/report`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  async scanPodcastAds(
    audioFile: File,
    onUploadProgress?: (percent: number) => void
  ): Promise<PodcastAdScanSubmitResponse> {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    const response = await this.client.post<PodcastAdScanSubmitResponse>(
      '/api/v1/audio-tools/podcast/scan-ads',
      formData,
      {
        onUploadProgress: (evt) => {
          if (evt.total != null && onUploadProgress) {
            onUploadProgress(Math.round((evt.loaded * 100) / evt.total));
          }
        },
      }
    );
    return response.data;
  }

  async getPodcastAdScanStatus(jobId: string): Promise<PodcastAdScanStatusResponse> {
    const response = await this.client.get<PodcastAdScanStatusResponse>(
      `/api/v1/audio-tools/podcast/scan-ads/${jobId}/status`
    );
    return response.data;
  }

  async exportPodcastAdAudio(jobId: string, exportMode: 'clean' | 'ads_only'): Promise<PodcastAdExportResponse> {
    const response = await this.client.post<PodcastAdExportResponse>('/api/v1/audio-tools/podcast/export', {
      job_id: jobId,
      export_mode: exportMode,
    });
    return response.data;
  }

  async downloadAudioToolsExport(filename: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/audio-tools/podcast/download/${encodeURIComponent(filename)}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async downloadSpeakerIsolationClip(jobId: string, filename: string): Promise<Blob> {
    const response = await this.client.get(
      `/api/v1/audio-tools/isolate-speakers/clip/${encodeURIComponent(jobId)}/${encodeURIComponent(filename)}`,
      { responseType: 'blob' }
    );
    return response.data;
  }

  async uploadForSpeakerIsolation(
    audioFile: File,
    onUploadProgress?: (percent: number) => void
  ): Promise<SpeakerIsolationSubmitResponse> {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    const response = await this.client.post<SpeakerIsolationSubmitResponse>(
      '/api/v1/audio-tools/isolate-speakers',
      formData,
      {
        onUploadProgress: (evt) => {
          if (evt.total != null && onUploadProgress) {
            onUploadProgress(Math.round((evt.loaded * 100) / evt.total));
          }
        },
      }
    );
    return response.data;
  }

  async getSpeakerIsolationStatus(jobId: string): Promise<SpeakerIsolationStatusResponse> {
    const response = await this.client.get<SpeakerIsolationStatusResponse>(
      `/api/v1/audio-tools/isolate-speakers/${encodeURIComponent(jobId)}/status`
    );
    return response.data;
  }

  async createVoiceFromIsolationClip(
    jobId: string,
    clipId: string,
    voiceName: string,
    voiceDescription?: string
  ): Promise<VoiceCreateResponse> {
    const response = await this.client.post<VoiceCreateResponse>(
      '/api/v1/audio-tools/isolate-speakers/create-voice',
      {
        job_id: jobId,
        clip_id: clipId,
        voice_name: voiceName,
        voice_description: voiceDescription || null,
      }
    );
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();