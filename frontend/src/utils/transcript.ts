import type { TranscriptSpeaker, TranscriptSpeakerSegment } from '../types/api';

const SENTENCE_END = /[.!?]\s*$/;

/**
 * Merges consecutive transcript segments from the same speaker into
 * sentence-level chunks. Reduces fragmentation from Whisper's phrase-level output.
 */
export function mergeTranscriptSegments(
  segments: TranscriptSpeakerSegment[]
): TranscriptSpeakerSegment[] {
  if (!segments.length) return [];

  const merged: TranscriptSpeakerSegment[] = [];
  let acc: TranscriptSpeakerSegment | null = null;

  for (const seg of segments) {
    const text = (seg.text || '').trim();
    if (!text) continue;

    const sameSpeaker = acc !== null && acc.speaker_id === seg.speaker_id;
    const endsSentence = SENTENCE_END.test(text);

    if (acc !== null && sameSpeaker) {
      acc = {
        speaker_id: acc.speaker_id,
        start_ms: acc.start_ms,
        text: `${acc.text} ${text}`.trim(),
        end_ms: seg.end_ms,
        confidence: (acc.confidence + seg.confidence) / 2,
      };
      if (endsSentence) {
        merged.push(acc);
        acc = null;
      }
    } else {
      if (acc) merged.push(acc);
      acc = { ...seg, text };
    }
  }
  if (acc) merged.push(acc);

  return merged;
}

/**
 * Converts transcript segments to speech-generate format:
 * "Speaker 1: ...", "Speaker 2: ..."
 */
export function transcriptToGenerateFormat(
  segments: TranscriptSpeakerSegment[],
  speakers: TranscriptSpeaker[]
): string {
  if (!segments.length) return '';

  const ordered = mergeTranscriptSegments(segments);
  const speakerIndexMap = new Map<string, number>();
  speakers.forEach((speaker, index) => {
    speakerIndexMap.set(speaker.id, index);
  });

  const lines = ordered
    .map((segment) => {
      const text = (segment.text || '').trim();
      if (!text) return null;

      const speakerIndex = speakerIndexMap.get(segment.speaker_id);
      const labelIndex = speakerIndex ?? 0;
      return `Speaker ${labelIndex + 1}: ${text}`;
    })
    .filter((line): line is string => Boolean(line));

  return lines.join('\n');
}
