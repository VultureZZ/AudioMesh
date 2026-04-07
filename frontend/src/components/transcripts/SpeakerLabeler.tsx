import { useMemo, useState } from 'react';
import type { TranscriptSpeaker } from '../../types/api';
import { Input } from '../Input';
import { Button } from '../Button';

interface SpeakerLabelerProps {
  speakers: TranscriptSpeaker[];
  onProceed: (labels: { id: string; label: string }[]) => Promise<void>;
  isLoading: boolean;
}

export function SpeakerLabeler({ speakers, onProceed, isLoading }: SpeakerLabelerProps) {
  const [labels, setLabels] = useState<Record<string, string>>({});

  const effectiveSpeakers = useMemo(() => speakers || [], [speakers]);
  if (effectiveSpeakers.length <= 1) return null;

  const handleProceed = async () => {
    const payload = effectiveSpeakers.map((s) => ({
      id: s.id,
      label: labels[s.id] ?? s.label ?? s.id,
    }));
    await onProceed(payload);
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Label speakers</h3>
      {effectiveSpeakers.map((speaker) => (
        <div key={speaker.id} className="border rounded-md p-4">
          <div className="text-sm text-gray-600 mb-2">
            {speaker.id} • {speaker.segment_count} segments • {Math.round(speaker.talk_time_seconds)}s talk time
          </div>
          <Input
            label="Name"
            value={labels[speaker.id] ?? speaker.label ?? ''}
            onChange={(e) => setLabels((prev) => ({ ...prev, [speaker.id]: e.target.value }))}
            placeholder="Enter speaker name"
          />
        </div>
      ))}
      <Button onClick={handleProceed} isLoading={isLoading}>
        Proceed to Analysis
      </Button>
    </div>
  );
}

