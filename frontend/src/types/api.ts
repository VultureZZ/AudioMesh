/**
 * TypeScript interfaces matching API schemas
 */

export interface SpeechSettings {
  language: string;
  output_format: string;
  sample_rate: number;
}

export interface SpeechGenerateRequest {
  transcript: string;
  speakers: string[];
  speaker_instructions?: string[];
  settings?: SpeechSettings;
}

export interface SpeechGenerateResponse {
  success: boolean;
  message: string;
  audio_url?: string;
  file_path?: string;
}

export interface VoiceQualityAnalysis {
  clone_quality: string;
  issues: string[];
  recording_quality_score: number;
  background_music_detected: boolean;
  background_noise_detected: boolean;
}

export interface VoiceResponse {
  id: string;
  name: string;
  display_name?: string;
  language_code?: string;
  language_label?: string;
  gender?: 'male' | 'female' | 'neutral' | 'unknown' | string;
  description?: string;
  type: string;
  created_at?: string;
  audio_files?: string[];
  image_url?: string;
  quality_analysis?: VoiceQualityAnalysis;
}

export interface VoiceListResponse {
  voices: VoiceResponse[];
  total: number;
}

export interface IndividualFileAnalysis {
  filename: string;
  duration_seconds?: number;
  sample_rate?: number;
  channels?: number;
  file_size_bytes?: number;
  file_size_mb?: number;
  warnings: string[];
  error?: string;
}

export interface AudioValidationFeedback {
  total_duration_seconds: number;
  individual_files: IndividualFileAnalysis[];
  warnings: string[];
  recommendations: string[];
  quality_metrics: Record<string, unknown>;
}

export interface VoiceCreateResponse {
  success: boolean;
  message: string;
  voice?: VoiceResponse;
  validation_feedback?: AudioValidationFeedback;
}

export interface AudioClipRange {
  start_seconds: number;
  end_seconds: number;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
}

export interface PodcastScriptRequest {
  url: string;
  voices: string[];
  genre: string;
  duration: string;
  ollama_url?: string;
  ollama_model?: string;
}

export interface PodcastScriptResponse {
  success: boolean;
  message: string;
  script?: string;
  warnings?: string[];
}

export interface PodcastGenerateRequest {
  script: string;
  voices: string[];
  settings?: SpeechSettings;
  title?: string;
  source_url?: string;
  genre?: string;
  duration?: string;
  save_to_library?: boolean;
}

export interface PodcastGenerateResponse {
  success: boolean;
  message: string;
  audio_url?: string;
  file_path?: string;
  script?: string;
  podcast_id?: string;
  warnings?: string[];
}

export interface PodcastItem {
  id: string;
  title: string;
  voices: string[];
  source_url?: string;
  genre?: string;
  duration?: string;
  created_at?: string;
  audio_url?: string;
}

export interface PodcastListResponse {
  podcasts: PodcastItem[];
  total: number;
}

export interface MusicGenerateRequest {
  caption?: string;
  lyrics?: string;
  bpm?: number;
  keyscale?: string;
  timesignature?: string;
  duration?: number;
  vocal_language?: string;
  instrumental?: boolean;
  thinking?: boolean;
  inference_steps?: number;
  batch_size?: number;
  seed?: number;
  audio_format?: 'mp3' | 'wav' | 'flac';
}

export interface MusicCoverGenerateParams {
  prompt?: string;
  lyrics?: string;
  duration?: number;
  audio_cover_strength?: number;
  vocal_language?: string;
  instrumental?: boolean;
  thinking?: boolean;
  inference_steps?: number;
  batch_size?: number;
  seed?: number;
  audio_format?: 'mp3' | 'wav' | 'flac';
}

export interface MusicGenerateResponse {
  success: boolean;
  message: string;
  task_id: string;
}

export interface MusicStatusMetadataItem {
  filename: string;
  audio_url: string;
  file_path: string;
  seed_value?: string;
  prompt?: string;
  lyrics?: string;
  metas?: Record<string, unknown>;
  dit_model?: string;
  lm_model?: string;
}

export interface MusicStatusResponse {
  success: boolean;
  message: string;
  task_id: string;
  status: 'running' | 'succeeded' | 'failed' | string;
  audios: string[];
  metadata: MusicStatusMetadataItem[];
  error?: string;
}

export interface MusicLyricsRequest {
  description: string;
  genre?: string;
  mood?: string;
  language?: string;
  duration_hint?: string;
}

export interface MusicLyricsResponse {
  success: boolean;
  message: string;
  lyrics: string;
  caption: string;
}

export interface MusicSimpleRequest {
  description: string;
  input_mode?: 'refine' | 'exact';
  instrumental?: boolean;
  vocal_language?: string;
  duration?: number;
  batch_size?: number;
  exact_caption?: string;
  exact_lyrics?: string;
  exact_bpm?: number;
  exact_keyscale?: string;
  exact_timesignature?: string;
}

export interface MusicHealthResponse {
  available: boolean;
  running: boolean;
  service: string;
  host: string;
  port: number;
}

export interface MusicPreset {
  id: string;
  name: string;
  mode: 'simple' | 'custom' | string;
  values: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface MusicPresetRequest {
  name: string;
  mode: 'simple' | 'custom' | string;
  values: Record<string, unknown>;
}

export interface MusicPresetListResponse {
  presets: MusicPreset[];
  total: number;
}

export interface MusicHistoryItem {
  id: string;
  task_id: string;
  mode: 'simple' | 'custom' | string;
  status: 'running' | 'succeeded' | 'failed' | string;
  request_payload: Record<string, unknown>;
  audios: string[];
  metadata: MusicStatusMetadataItem[];
  error?: string;
  created_at?: string;
  updated_at?: string;
}

export interface MusicHistoryListResponse {
  history: MusicHistoryItem[];
  total: number;
}

export interface VoiceProfile {
  cadence?: string;
  tone?: string;
  vocabulary_style?: string;
  sentence_structure?: string;
  unique_phrases: string[];
  keywords: string[];
  profile_text?: string;
  created_at?: string;
  updated_at?: string;
}

export interface VoiceProfileApplyRequest {
  cadence?: string;
  tone?: string;
  vocabulary_style?: string;
  sentence_structure?: string;
  unique_phrases?: string[];
  keywords?: string[];
  profile_text?: string;
}

export interface VoiceProfileRequest {
  keywords?: string[];
  ollama_url?: string;
  ollama_model?: string;
}

export interface VoiceProfileResponse {
  success: boolean;
  message: string;
  profile?: VoiceProfile;
}

export interface VoiceProfileFromAudioResponse {
  success: boolean;
  message: string;
  profile?: VoiceProfile;
  transcript?: string;
  validation_feedback?: AudioValidationFeedback;
}

export interface VoiceUpdateRequest {
  name?: string;
  description?: string;
  language_code?: string;
  gender?: string;
}

export interface VoiceUpdateResponse {
  success: boolean;
  message: string;
  voice?: VoiceResponse;
}

export type RecordingType = 'meeting' | 'call' | 'memo' | 'interview' | 'other';

export interface TranscriptSpeakerSegment {
  speaker_id: string;
  start_ms: number;
  end_ms: number;
  text: string;
  confidence: number;
}

export interface TranscriptSpeaker {
  id: string;
  label?: string | null;
  voice_library_match?: string | null;
  match_confidence?: number | null;
  talk_time_seconds: number;
  segment_count: number;
  summary?: string | null;
  audio_segment_path?: string | null;
}

export interface TranscriptActionItem {
  action: string;
  owner?: string | null;
  due_hint?: string | null;
  priority: 'low' | 'medium' | 'high' | string;
}

export interface TranscriptAnalysis {
  summary: string;
  action_items: TranscriptActionItem[];
  key_decisions: string[];
  open_questions: string[];
  topics_discussed: string[];
  sentiment: string;
  duration_formatted: string;
}

export interface TranscriptItem {
  id: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
  duration_seconds?: number;
  file_name: string;
  file_size_bytes: number;
  language: string;
  recording_type: RecordingType;
  speakers: TranscriptSpeaker[];
  transcript: TranscriptSpeakerSegment[];
  analysis?: TranscriptAnalysis | null;
  error?: string | null;
  progress_pct: number;
  current_stage?: string | null;
}

export interface TranscriptUploadResponse {
  transcript_id: string;
  status: string;
  message: string;
  estimated_wait_seconds: number;
}

export interface TranscriptStatusResponse {
  transcript_id: string;
  status: string;
  progress_pct: number;
  current_stage?: string | null;
  duration_seconds?: number | null;
  speakers_detected?: number | null;
  error?: string | null;
}

export interface TranscriptListResponse {
  transcripts: TranscriptItem[];
  total: number;
  limit: number;
  offset: number;
}