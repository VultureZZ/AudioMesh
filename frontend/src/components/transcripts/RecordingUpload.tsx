import { useState } from 'react';
import { Button } from '../Button';
import { Input } from '../Input';
import { Select } from '../Select';
import type { RecordingType } from '../../types/api';

interface RecordingUploadProps {
  isLoading: boolean;
  onSubmit: (payload: { file: File; title?: string; recordingType: RecordingType; language: string }) => Promise<void>;
}

export function RecordingUpload({ isLoading, onSubmit }: RecordingUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [language, setLanguage] = useState('en');
  const [recordingType, setRecordingType] = useState<RecordingType>('meeting');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    await onSubmit({ file, title: title || undefined, recordingType, language });
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6 space-y-4">
      <h2 className="text-xl font-semibold text-gray-900">Upload recording</h2>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Audio file<span className="text-red-500 ml-1">*</span>
        </label>
        <input
          type="file"
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
          accept=".mp3,.wav,.m4a,.mp4,.webm,.ogg,.flac,audio/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          required
        />
      </div>
      <Input
        label="Title (optional)"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="e.g., Team sync, Client call, Voice memo"
      />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Select
          label="Recording type"
          value={recordingType}
          onChange={(e) => setRecordingType(e.target.value as RecordingType)}
          options={[
            { value: 'meeting', label: 'Meeting' },
            { value: 'call', label: 'Call' },
            { value: 'memo', label: 'Voice memo / self-note' },
            { value: 'interview', label: 'Interview' },
            { value: 'other', label: 'Other' },
          ]}
        />
        <Input
          label="Language code"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          placeholder="en"
        />
      </div>
      <div className="text-sm text-gray-500">
        Supported formats: MP3, WAV, M4A, MP4, WEBM, OGG, FLAC
      </div>
      <Button type="submit" isLoading={isLoading} disabled={!file}>
        Upload and Process
      </Button>
    </form>
  );
}

