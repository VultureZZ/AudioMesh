/**
 * Typed helpers for Audio Tools (podcast ad scanner) endpoints.
 * Uses the shared API client so API key and base URL stay in sync with Settings.
 */

import { apiClient } from '../services/api';
import type {
  PodcastAdExportResponse,
  PodcastAdScanStatusResponse,
  PodcastAdScanSubmitResponse,
  SpeakerIsolationStatusResponse,
  SpeakerIsolationSubmitResponse,
  VoiceCreateResponse,
} from '../types/api';

export type {
  AdSegmentItem,
  PodcastAdExportResponse,
  PodcastAdScanStatusResponse,
  PodcastAdScanSubmitResponse,
  SpeakerIsolationClipItem,
  SpeakerIsolationSpeakerItem,
  SpeakerIsolationStatusResponse,
  SpeakerIsolationSubmitResponse,
  VoiceCreateResponse,
} from '../types/api';

export function scanPodcastAds(
  audioFile: File,
  onUploadProgress?: (percent: number) => void
): Promise<PodcastAdScanSubmitResponse> {
  return apiClient.scanPodcastAds(audioFile, onUploadProgress);
}

export function getPodcastAdScanStatus(jobId: string): Promise<PodcastAdScanStatusResponse> {
  return apiClient.getPodcastAdScanStatus(jobId);
}

export function exportPodcastAdAudio(
  jobId: string,
  exportMode: 'clean' | 'ads_only'
): Promise<PodcastAdExportResponse> {
  return apiClient.exportPodcastAdAudio(jobId, exportMode);
}

export function downloadAudioToolsExport(filename: string): Promise<Blob> {
  return apiClient.downloadAudioToolsExport(filename);
}

export function uploadForSpeakerIsolation(
  file: File,
  onUploadProgress?: (percent: number) => void
): Promise<SpeakerIsolationSubmitResponse> {
  return apiClient.uploadForSpeakerIsolation(file, onUploadProgress);
}

export function getSpeakerIsolationStatus(jobId: string): Promise<SpeakerIsolationStatusResponse> {
  return apiClient.getSpeakerIsolationStatus(jobId);
}

export function createVoiceFromClip(
  jobId: string,
  clipId: string,
  voiceName: string,
  description?: string
): Promise<VoiceCreateResponse> {
  return apiClient.createVoiceFromIsolationClip(jobId, clipId, voiceName, description);
}
