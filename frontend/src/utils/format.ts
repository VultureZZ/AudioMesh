/**
 * Formatting utility functions
 */
import type { VoiceResponse } from '../types/api';
import { getLanguageLabel } from './languages';

/**
 * Format duration in seconds to mm:ss format
 */
export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format file size in bytes to human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format date/time string
 */
export function formatDateTime(dateString: string | undefined): string {
  if (!dateString) return 'N/A';
  
  try {
    const date = new Date(dateString);
    return date.toLocaleString();
  } catch {
    return dateString;
  }
}

/**
 * Format date only
 */
export function formatDate(dateString: string | undefined): string {
  if (!dateString) return 'N/A';
  
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  } catch {
    return dateString;
  }
}

function getGenderIcon(gender: string | undefined): string {
  switch ((gender || '').toLowerCase()) {
    case 'female':
      return '♀';
    case 'male':
      return '♂';
    case 'neutral':
      return '⚧';
    default:
      return '';
  }
}

export function getVoiceDisplayName(voice: Pick<VoiceResponse, 'name' | 'display_name'>): string {
  return voice.display_name || voice.name;
}

export interface FormatVoiceLabelOptions {
  showQuality?: boolean;
}

export function formatVoiceLabel(
  voice: Pick<VoiceResponse, 'name' | 'display_name' | 'language_code' | 'language_label' | 'gender' | 'quality_analysis'>,
  options?: FormatVoiceLabelOptions
): string {
  const displayName = getVoiceDisplayName(voice);
  const genderIcon = getGenderIcon(voice.gender);
  const languageLabel =
    voice.language_label ||
    (voice.language_code ? getLanguageLabel(voice.language_code) : '');

  const suffixParts: string[] = [];
  if (genderIcon) suffixParts.push(genderIcon);
  if (languageLabel) suffixParts.push(`(${languageLabel})`);

  let label = suffixParts.length ? `${displayName} ${suffixParts.join(' ')}` : displayName;

  if (options?.showQuality && voice.quality_analysis) {
    const qa = voice.quality_analysis;
    const issueLabels = getIssueDisplayLabels(qa.issues);
    if (issueLabels.length > 0) {
      label += ` - ${issueLabels.join(', ')}`;
    } else {
      label += ` - ${getQualityDisplayLabel(qa.clone_quality)}`;
    }
  }

  return label;
}

export function getQualityDisplayLabel(cloneQuality: string): string {
  const labels: Record<string, string> = {
    excellent: 'Excellent',
    good: 'Good',
    fair: 'Fair',
    poor: 'Poor',
  };
  return labels[cloneQuality] || cloneQuality;
}

export function getIssueDisplayLabels(issues: string[]): string[] {
  const labels: Record<string, string> = {
    background_music: 'Background music',
    background_noise: 'Background noise',
    low_recording_quality: 'Low recording quality',
    background_audio: 'Background audio',
  };
  return issues.map((i) => labels[i] || i);
}