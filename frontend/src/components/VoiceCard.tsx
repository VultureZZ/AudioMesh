/**
 * Card component for displaying voice information
 */

import { VoiceResponse } from '../types/api';
import {
  formatDate,
  formatVoiceLabel,
  getVoiceDisplayName,
  getQualityDisplayLabel,
  getIssueDisplayLabels,
} from '../utils/format';
import { Button } from './Button';

interface VoiceCardProps {
  voice: VoiceResponse;
  apiBaseUrl?: string;
  onDelete?: (voiceId: string) => void;
  onEdit?: (voiceId: string) => void;
  onViewProfile?: (voiceId: string) => void;
  isDeleting?: boolean;
  hasProfile?: boolean;
}

export function VoiceCard({
  voice,
  apiBaseUrl = '',
  onDelete,
  onEdit,
  onViewProfile,
  isDeleting,
  hasProfile,
}: VoiceCardProps) {
  const isCustom =
    voice.type === 'custom' || voice.type === 'voice_design';
  const displayName = getVoiceDisplayName(voice);
  const formattedLabel = formatVoiceLabel(voice);
  const suffix = formattedLabel.startsWith(displayName) ? formattedLabel.slice(displayName.length).trim() : '';
  const imageSrc =
    voice.image_url && apiBaseUrl ? `${apiBaseUrl.replace(/\/$/, '')}${voice.image_url}` : null;
  const initial = displayName.charAt(0).toUpperCase() || '?';

  return (
    <div className="border rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow flex flex-col h-full">
      <div className="flex gap-4 flex-1">
        <div className="flex-shrink-0">
          {imageSrc ? (
            <img
              src={imageSrc}
              alt={displayName}
              className="w-16 h-16 rounded-full object-cover border border-gray-200"
            />
          ) : (
            <div
              className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center text-xl font-semibold text-gray-600 border border-gray-300"
              aria-hidden
            >
              {initial}
            </div>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <h3 className="text-lg font-semibold text-gray-900 truncate">
              {displayName}
              {suffix ? (
                <span className="ml-1 text-sm font-normal text-gray-600">{suffix}</span>
              ) : null}
            </h3>
            <span
              className={`px-2 py-0.5 text-xs font-medium rounded ${
                isCustom ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
              }`}
            >
              {voice.type}
            </span>
            {hasProfile && (
              <span className="px-2 py-0.5 text-xs font-medium rounded bg-green-100 text-green-800">
                Profiled
              </span>
            )}
            {voice.quality_analysis && (
              <span
                className={`px-2 py-0.5 text-xs font-medium rounded ${
                  voice.quality_analysis.clone_quality === 'excellent'
                    ? 'bg-emerald-100 text-emerald-800'
                    : voice.quality_analysis.clone_quality === 'good'
                      ? 'bg-green-100 text-green-800'
                      : voice.quality_analysis.clone_quality === 'fair'
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-red-100 text-red-800'
                }`}
              >
                {getQualityDisplayLabel(voice.quality_analysis.clone_quality)}
              </span>
            )}
            {voice.quality_analysis?.issues && voice.quality_analysis.issues.length > 0 && (
              <>
                {getIssueDisplayLabels(voice.quality_analysis.issues).map((issueLabel) => (
                  <span
                    key={issueLabel}
                    className="px-2 py-0.5 text-xs font-medium rounded bg-orange-100 text-orange-800"
                    title={`Quality issue: ${issueLabel}`}
                  >
                    {issueLabel}
                  </span>
                ))}
              </>
            )}
          </div>
          {voice.description && (
            <p className="text-sm text-gray-600 mb-2 line-clamp-2">{voice.description}</p>
          )}
          <div className="flex flex-wrap gap-3 text-xs text-gray-500">
            <span className="truncate" title={voice.id}>
              ID: {voice.id}
            </span>
            {voice.created_at && <span>Created: {formatDate(voice.created_at)}</span>}
            {voice.audio_files && voice.audio_files.length > 0 && (
              <span>{voice.audio_files.length} file(s)</span>
            )}
          </div>
        </div>
        {isCustom && (
          <div className="flex flex-col gap-2 flex-shrink-0">
            <div className="flex gap-2">
              {onEdit && (
                <Button
                  variant="secondary"
                  onClick={() => onEdit(voice.id)}
                  disabled={isDeleting}
                  className="text-xs"
                >
                  Edit
                </Button>
              )}
              {onViewProfile && (
                <Button
                  variant="secondary"
                  onClick={() => onViewProfile(voice.id)}
                  disabled={isDeleting}
                  className="text-xs"
                >
                  Profile
                </Button>
              )}
            </div>
            {onDelete && (
              <Button
                variant="danger"
                onClick={() => onDelete(voice.id)}
                isLoading={isDeleting}
                className="text-xs"
              >
                Delete
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}