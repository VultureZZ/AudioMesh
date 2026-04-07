/**
 * Modal for analyzing an audio file to derive a voice profile, then applying it to a voice.
 */

import { useEffect, useMemo, useState } from 'react';
import type { VoiceProfile, VoiceProfileApplyRequest, VoiceProfileFromAudioResponse, VoiceResponse } from '../types/api';
import { Button } from './Button';
import { Input } from './Input';
import { Select } from './Select';
import { Alert } from './Alert';
import { LoadingSpinner } from './LoadingSpinner';
import { isValidAudioFile } from '../utils/validation';
import { formatVoiceLabel } from '../utils/format';

interface VoiceProfileFromAudioModalProps {
  isOpen: boolean;
  onClose: () => void;
  voices: VoiceResponse[];
  defaultOllamaUrl?: string;
  defaultOllamaModel?: string;
  onAnalyze: (
    audioFile: File,
    keywords?: string,
    ollamaUrl?: string,
    ollamaModel?: string
  ) => Promise<VoiceProfileFromAudioResponse | null>;
  onApply: (voiceId: string, profile: VoiceProfileApplyRequest) => Promise<boolean>;
}

export function VoiceProfileFromAudioModal({
  isOpen,
  onClose,
  voices,
  defaultOllamaUrl,
  defaultOllamaModel,
  onAnalyze,
  onApply,
}: VoiceProfileFromAudioModalProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [keywords, setKeywords] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState(defaultOllamaUrl || '');
  const [ollamaModel, setOllamaModel] = useState(defaultOllamaModel || '');

  const [targetVoiceId, setTargetVoiceId] = useState<string>('');

  const [analyzing, setAnalyzing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [validationSummary, setValidationSummary] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    // initialize selection
    if (!targetVoiceId && voices.length > 0) {
      setTargetVoiceId(voices[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, voices.length]);

  useEffect(() => {
    if (isOpen) {
      setOllamaUrl(defaultOllamaUrl || '');
      setOllamaModel(defaultOllamaModel || '');
    }
  }, [isOpen, defaultOllamaUrl, defaultOllamaModel]);

  const voiceOptions = useMemo(
    () =>
      (voices || []).map((v) => ({
        value: v.id,
        label: `${formatVoiceLabel(v)} (${v.type})`,
      })),
    [voices]
  );

  const handlePickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (!f) return;
    if (!isValidAudioFile(f)) {
      setError('Please choose a valid audio file');
      return;
    }
    setError(null);
    setAudioFile(f);
  };

  const handleAnalyze = async () => {
    if (!audioFile) {
      setError('Please choose an audio file');
      return;
    }

    setAnalyzing(true);
    setError(null);
    setSuccess(null);
    setProfile(null);
    setTranscript(null);
    setValidationSummary(null);

    const resp = await onAnalyze(
      audioFile,
      keywords.trim() || undefined,
      ollamaUrl.trim() || undefined,
      ollamaModel.trim() || undefined
    );

    setAnalyzing(false);
    if (!resp) return;

    if (!resp.profile) {
      setError(resp.message || 'No profile returned');
      return;
    }

    setProfile(resp.profile);
    setTranscript(resp.transcript || null);

    if (resp.validation_feedback) {
      const vf = resp.validation_feedback;
      const warnings = vf.warnings?.length ? `Warnings: ${vf.warnings.join(', ')}` : null;
      const recs = vf.recommendations?.length ? `Recommendations: ${vf.recommendations.join(', ')}` : null;
      setValidationSummary(
        [`Total duration: ${vf.total_duration_seconds.toFixed(2)}s`, warnings, recs].filter(Boolean).join('\n')
      );
    }

    setSuccess('Profile generated from audio. Review it, then apply to a voice.');
  };

  const handleApply = async () => {
    if (!profile) {
      setError('Please analyze an audio file first');
      return;
    }
    if (!targetVoiceId) {
      setError('Please select a target voice');
      return;
    }

    setApplying(true);
    setError(null);
    setSuccess(null);

    const payload: VoiceProfileApplyRequest = {
      cadence: profile.cadence,
      tone: profile.tone,
      vocabulary_style: profile.vocabulary_style,
      sentence_structure: profile.sentence_structure,
      unique_phrases: profile.unique_phrases || [],
      keywords: profile.keywords || [],
      profile_text: profile.profile_text,
    };

    const ok = await onApply(targetVoiceId, payload);
    setApplying(false);
    if (ok) {
      setSuccess('Profile applied successfully');
    }
  };

  const resetAndClose = () => {
    setAudioFile(null);
    setKeywords('');
    setProfile(null);
    setTranscript(null);
    setValidationSummary(null);
    setError(null);
    setSuccess(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">Analyze Audio → Voice Profile</h2>
          <Button variant="secondary" onClick={resetAndClose}>
            Close
          </Button>
        </div>

        {error && <Alert type="error" message={error} onClose={() => setError(null)} />}
        {success && <Alert type="success" message={success} onClose={() => setSuccess(null)} />}

        <div className="space-y-6">
          <div className="bg-gray-50 border rounded-lg p-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Audio file</label>
              <input type="file" accept="audio/*" onChange={handlePickFile} />
              {audioFile && <p className="mt-1 text-xs text-gray-600">{audioFile.name}</p>}
            </div>

            <Input
              label="Keywords (Optional, comma-separated)"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              placeholder="e.g., calm, technical, upbeat"
              helpText="Used as light context; the transcript drives the profile"
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input
                label="Ollama URL (Optional)"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
                placeholder="http://localhost:11434"
              />
              <Input
                label="Ollama Model (Optional)"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
                placeholder="llama3.2"
              />
            </div>

            <Button variant="primary" onClick={handleAnalyze} isLoading={analyzing} disabled={!audioFile}>
              Analyze Audio
            </Button>
          </div>

          {analyzing && (
            <div className="flex justify-center items-center py-6">
              <LoadingSpinner size="lg" />
            </div>
          )}

          {validationSummary && (
            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Audio validation</h3>
              <pre className="text-sm text-gray-700 whitespace-pre-wrap">{validationSummary}</pre>
            </div>
          )}

          {transcript && (
            <div className="bg-white border rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Transcript</h3>
              <pre className="text-sm text-gray-700 whitespace-pre-wrap">{transcript}</pre>
            </div>
          )}

          {profile && (
            <div className="bg-white border rounded-lg p-4 space-y-3">
              <h3 className="text-lg font-semibold text-gray-900">Derived profile</h3>
              {profile.cadence && (
                <div>
                  <div className="text-sm font-medium text-gray-700">Cadence</div>
                  <div className="text-sm text-gray-900">{profile.cadence}</div>
                </div>
              )}
              {profile.tone && (
                <div>
                  <div className="text-sm font-medium text-gray-700">Tone</div>
                  <div className="text-sm text-gray-900">{profile.tone}</div>
                </div>
              )}
              {profile.vocabulary_style && (
                <div>
                  <div className="text-sm font-medium text-gray-700">Vocabulary</div>
                  <div className="text-sm text-gray-900">{profile.vocabulary_style}</div>
                </div>
              )}
              {profile.sentence_structure && (
                <div>
                  <div className="text-sm font-medium text-gray-700">Sentence structure</div>
                  <div className="text-sm text-gray-900">{profile.sentence_structure}</div>
                </div>
              )}
              {profile.unique_phrases?.length ? (
                <div>
                  <div className="text-sm font-medium text-gray-700">Unique phrases</div>
                  <div className="text-sm text-gray-900">{profile.unique_phrases.join(', ')}</div>
                </div>
              ) : null}
              {profile.profile_text && (
                <div>
                  <div className="text-sm font-medium text-gray-700">Full description</div>
                  <div className="text-sm text-gray-900 whitespace-pre-wrap">{profile.profile_text}</div>
                </div>
              )}
            </div>
          )}

          <div className="bg-white border rounded-lg p-4 space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Apply profile to voice</h3>
            {voiceOptions.length > 0 ? (
              <Select
                label="Target voice"
                options={voiceOptions}
                value={targetVoiceId}
                onChange={(e) => setTargetVoiceId(e.target.value)}
              />
            ) : (
              <p className="text-sm text-gray-600">No voices available</p>
            )}

            <Button
              variant="primary"
              onClick={handleApply}
              isLoading={applying}
              disabled={!profile || !targetVoiceId}
            >
              Apply to Voice
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

