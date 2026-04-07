/**
 * Modal component for viewing and editing voice profiles
 */

import { useState, useEffect } from 'react';
import { VoiceProfile, VoiceProfileResponse } from '../types/api';
import { Button } from './Button';
import { Input } from './Input';
import { LoadingSpinner } from './LoadingSpinner';
import { Alert } from './Alert';

interface VoiceProfileModalProps {
  voiceId: string;
  voiceName: string;
  isOpen: boolean;
  onClose: () => void;
  onGetProfile: (voiceId: string) => Promise<VoiceProfileResponse | null>;
  onUpdateProfile: (voiceId: string, keywords: string[]) => Promise<VoiceProfileResponse | null>;
  onGenerateProfile?: (voiceId: string, keywords: string[]) => Promise<VoiceProfileResponse | null>;
}

export function VoiceProfileModal({
  voiceId,
  voiceName,
  isOpen,
  onClose,
  onGetProfile,
  onUpdateProfile,
  onGenerateProfile,
}: VoiceProfileModalProps) {
  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  const [keywords, setKeywords] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && voiceId) {
      loadProfile();
    } else {
      setProfile(null);
      setKeywords('');
      setError(null);
      setSuccess(null);
    }
  }, [isOpen, voiceId]);

  const loadProfile = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await onGetProfile(voiceId);
      if (response && response.profile) {
        setProfile(response.profile);
        setKeywords(response.profile.keywords?.join(', ') || '');
      } else {
        setProfile(null);
        setKeywords('');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load profile');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveKeywords = async () => {
    if (!keywords.trim()) {
      setError('Please enter at least one keyword');
      return;
    }

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const keywordsList = keywords
        .split(',')
        .map((k) => k.trim())
        .filter((k) => k.length > 0);

      const response = await onUpdateProfile(voiceId, keywordsList);
      if (response && response.profile) {
        setProfile(response.profile);
        setSuccess('Profile updated successfully');
        // Reload profile to get updated data
        await loadProfile();
      } else {
        setError('Failed to update profile');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update profile');
    } finally {
      setSaving(false);
    }
  };

  const handleGenerateProfile = async () => {
    if (!onGenerateProfile) {
      setError('Profile generation not available');
      return;
    }

    setGenerating(true);
    setError(null);
    setSuccess(null);

    try {
      const keywordsList = keywords
        .split(',')
        .map((k) => k.trim())
        .filter((k) => k.length > 0);

      const response = await onGenerateProfile(voiceId, keywordsList.length > 0 ? keywordsList : []);
      if (response && response.profile) {
        setProfile(response.profile);
        setSuccess('Profile generated successfully');
        // Reload profile to get updated data
        await loadProfile();
      } else {
        setError('Failed to generate profile');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate profile');
    } finally {
      setGenerating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-semibold text-gray-900">
            Voice Profile: {voiceName}
          </h2>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>

        {error && (
          <Alert type="error" message={error} onClose={() => setError(null)} />
        )}
        {success && (
          <Alert type="success" message={success} onClose={() => setSuccess(null)} />
        )}

        {loading ? (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner size="lg" />
          </div>
        ) : profile ? (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Profile Details</h3>
              <div className="space-y-3">
                {profile.cadence && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Cadence</label>
                    <p className="mt-1 text-sm text-gray-900">{profile.cadence}</p>
                  </div>
                )}
                {profile.tone && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Tone</label>
                    <p className="mt-1 text-sm text-gray-900">{profile.tone}</p>
                  </div>
                )}
                {profile.vocabulary_style && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Vocabulary Style</label>
                    <p className="mt-1 text-sm text-gray-900">{profile.vocabulary_style}</p>
                  </div>
                )}
                {profile.sentence_structure && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Sentence Structure</label>
                    <p className="mt-1 text-sm text-gray-900">{profile.sentence_structure}</p>
                  </div>
                )}
                {profile.unique_phrases && profile.unique_phrases.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Unique Phrases</label>
                    <p className="mt-1 text-sm text-gray-900">
                      {profile.unique_phrases.join(', ')}
                    </p>
                  </div>
                )}
                {profile.profile_text && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Full Description</label>
                    <p className="mt-1 text-sm text-gray-900 whitespace-pre-wrap">
                      {profile.profile_text}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="border-t pt-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Keywords</h3>
              <Input
                label="Keywords (comma-separated)"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                placeholder="e.g., Donald Trump, President, Politician"
                helpText="Enter keywords to enhance the voice profile. These help identify unique speech patterns."
              />
              <div className="flex gap-3 mt-4">
                {onGenerateProfile && (
                  <Button
                    variant="primary"
                    onClick={handleGenerateProfile}
                    isLoading={generating}
                    className="flex-1"
                  >
                    Regenerate Profile
                  </Button>
                )}
                <Button
                  variant="secondary"
                  onClick={handleSaveKeywords}
                  isLoading={saving}
                  className="flex-1"
                >
                  Update Keywords
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-500 mb-4">No profile found for this voice.</p>
            <div className="border-t pt-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Create Profile</h3>
              <Input
                label="Keywords (comma-separated)"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                placeholder="e.g., Donald Trump, President, Politician"
                helpText="Enter keywords to create a voice profile. These help identify unique speech patterns."
              />
              <div className="flex gap-3 mt-4">
                {onGenerateProfile && (
                  <Button
                    variant="primary"
                    onClick={handleGenerateProfile}
                    isLoading={generating}
                    className="flex-1"
                  >
                    Generate Profile
                  </Button>
                )}
                <Button
                  variant="secondary"
                  onClick={handleSaveKeywords}
                  isLoading={saving}
                  className="flex-1"
                >
                  Create with Keywords
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
