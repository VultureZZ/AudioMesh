import type { TranscriptSpeakerSegment } from '../../types/api';
import { mergeTranscriptSegments } from '../../utils/transcript';

interface TranscriptViewerProps {
  transcript: TranscriptSpeakerSegment[];
}

export function TranscriptViewer({ transcript }: TranscriptViewerProps) {
  const merged = mergeTranscriptSegments(transcript ?? []);
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-3">
      <h3 className="text-lg font-semibold text-gray-900">Transcript</h3>
      <div className="max-h-[440px] overflow-y-auto space-y-2">
        {merged.map((seg, idx) => (
          <div key={`${seg.speaker_id}-${seg.start_ms}-${idx}`} className="text-sm">
            <span className="font-medium text-primary-700">{seg.speaker_id}</span>
            <span className="text-gray-400"> [{Math.round(seg.start_ms / 1000)}s]</span>
            <span className="text-gray-800"> {seg.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

