/**
 * Voice management interface
 */

import { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';
import { useVoices } from '../hooks/useVoices';
import { useSettings } from '../hooks/useSettings';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { VoiceCard } from '../components/VoiceCard';
import { VoiceProfileModal } from '../components/VoiceProfileModal';
import { VoiceProfileFromAudioModal } from '../components/VoiceProfileFromAudioModal';
import { CreateVoiceModal } from '../components/CreateVoiceModal';
import { Alert } from '../components/Alert';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { Select } from '../components/Select';
import { SUPPORTED_LANGUAGES } from '../utils/languages';

export function VoicesPage() {
  const { voices, loading: voicesLoading, refresh } = useVoices();
  const { settings } = useSettings();
  const {
    createVoiceUnified,
    deleteVoice,
    updateVoice,
    uploadVoiceImage,
    getVoiceProfile,
    createOrUpdateVoiceProfile,
    updateVoiceProfileKeywords,
    generateVoiceProfile,
    analyzeVoiceProfileFromAudio,
    applyVoiceProfile,
    loading: apiLoading,
    error: apiError,
  } = useApi();

  const [createVoiceModalOpen, setCreateVoiceModalOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editLanguageCode, setEditLanguageCode] = useState<string>('');
  const [editGender, setEditGender] = useState<string>('unknown');
  const [editImage, setEditImage] = useState<File | null>(null);
  const [profileModalVoiceId, setProfileModalVoiceId] = useState<string | null>(null);
  const [profileFromAudioOpen, setProfileFromAudioOpen] = useState(false);
  const [voiceProfiles, setVoiceProfiles] = useState<Record<string, boolean>>({});
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [validationFeedback, setValidationFeedback] = useState<string | null>(null);

  const handleCreateVoiceUnified = async (
    params: Parameters<typeof createVoiceUnified>[0]
  ): Promise<import('../types/api').VoiceCreateResponse | null> => {
    setErrorMessage(null);
    setSuccessMessage(null);
    setValidationFeedback(null);

    const response = await createVoiceUnified(params);
    if (response) {
      if (response.success) {
        setSuccessMessage(response.message);
        refresh();
        if (response.validation_feedback) {
          const feedback = response.validation_feedback;
          const warnings =
            feedback.warnings.length > 0 ? `\nWarnings: ${feedback.warnings.join(', ')}` : '';
          const recommendations =
            feedback.recommendations.length > 0
              ? `\nRecommendations: ${feedback.recommendations.join(', ')}`
              : '';
          setValidationFeedback(
            `Total duration: ${feedback.total_duration_seconds.toFixed(2)}s${warnings}${recommendations}`
          );
        }
      } else {
        setErrorMessage(response.message || 'Failed to create voice');
      }
    }
    return response;
  };

  const handleDeleteVoice = async (voiceId: string) => {
    if (!confirm('Are you sure you want to delete this voice? This action cannot be undone.')) {
      return;
    }

    setDeletingId(voiceId);
    const success = await deleteVoice(voiceId);
    setDeletingId(null);

    if (success) {
      setSuccessMessage('Voice deleted successfully');
      refresh();
    }
  };

  const handleEditVoice = (voiceId: string) => {
    const voice = voices.find((v) => v.id === voiceId);
    if (voice) {
      setEditingId(voiceId);
      setEditName(voice.name);
      setEditDescription(voice.description || '');
      setEditLanguageCode(voice.language_code || '');
      setEditGender((voice.gender as string) || 'unknown');
      setEditImage(null);
    }
  };

  const handleSaveEdit = async () => {
    if (!editingId) return;

    const response = await updateVoice(editingId, {
      name: editName.trim() || undefined,
      description: editDescription.trim() || undefined,
      language_code: editLanguageCode,
      gender: editGender,
    });

    if (response && response.success) {
      if (editImage) {
        const imageResponse = await uploadVoiceImage(editingId, editImage);
        if (imageResponse && imageResponse.success) {
          setSuccessMessage('Voice and image updated successfully');
        } else {
          setSuccessMessage('Voice updated; image upload failed');
        }
      } else {
        setSuccessMessage('Voice updated successfully');
      }
      setEditingId(null);
      setEditName('');
      setEditDescription('');
      setEditLanguageCode('');
      setEditGender('unknown');
      setEditImage(null);
      refresh();
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName('');
    setEditDescription('');
    setEditLanguageCode('');
    setEditGender('unknown');
    setEditImage(null);
  };

  // Load profile status for custom voices
  useEffect(() => {
    const loadProfiles = async () => {
      const profiles: Record<string, boolean> = {};
      for (const voice of voices) {
        try {
          const response = await getVoiceProfile(voice.id);
          profiles[voice.id] = !!(response && response.profile);
        } catch {
          profiles[voice.id] = false;
        }
      }
      setVoiceProfiles(profiles);
    };

    if (voices.length > 0) {
      loadProfiles();
    }
  }, [voices, getVoiceProfile]);

  const handleViewProfile = (voiceId: string) => {
    setProfileModalVoiceId(voiceId);
  };

  const handleApplyProfileFromAudio = async (voiceId: string, profile: import('../types/api').VoiceProfileApplyRequest) => {
    const resp = await applyVoiceProfile(voiceId, profile);
    if (resp && resp.success) {
      setVoiceProfiles((prev) => ({ ...prev, [voiceId]: true }));
      setSuccessMessage('Profile applied successfully');
      return true;
    }
    return false;
  };

  const handleUpdateProfileKeywords = async (voiceId: string, keywords: string[]) => {
    // Try to update keywords first, if that fails, create profile
    let response = await updateVoiceProfileKeywords(voiceId, {
      keywords,
      ollama_url: settings.ollamaServerUrl,
      ollama_model: settings.ollamaModel,
    });
    if (!response || !response.profile) {
      // If update failed, try creating profile
      response = await createOrUpdateVoiceProfile(voiceId, {
        keywords,
        ollama_url: settings.ollamaServerUrl,
        ollama_model: settings.ollamaModel,
      });
    }
    if (response && response.profile) {
      setVoiceProfiles((prev) => ({ ...prev, [voiceId]: true }));
      return response;
    }
    return null;
  };

  const handleGenerateProfile = async (voiceId: string, keywords: string[]) => {
    const response = await generateVoiceProfile(voiceId, {
      keywords: keywords.length > 0 ? keywords : undefined,
      ollama_url: settings.ollamaServerUrl,
      ollama_model: settings.ollamaModel,
    });
    if (response && response.profile) {
      setVoiceProfiles((prev) => ({ ...prev, [voiceId]: true }));
      return response;
    }
    return null;
  };

  const customVoices = voices.filter(
    (v) => v.type === 'custom' || v.type === 'voice_design'
  );
  const defaultVoices = voices.filter((v) => v.type === 'default');

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Voices</h1>
          <p className="mt-2 text-gray-600">
            Create and manage custom voices from audio, or use built-in default voices. Add an optional avatar image to represent each voice.
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={() => setProfileFromAudioOpen(true)}>
            Analyze Audio → Profile
          </Button>
          <Button variant="primary" onClick={() => setCreateVoiceModalOpen(true)}>
            Create Voice
          </Button>
        </div>
      </div>

      {apiError && <Alert type="error" message={apiError} />}
      {errorMessage && <Alert type="error" message={errorMessage} />}
      {successMessage && (
        <Alert
          type="success"
          message={successMessage}
          onClose={() => setSuccessMessage(null)}
        />
      )}
      {validationFeedback && (
        <Alert
          type="info"
          message={validationFeedback}
          onClose={() => setValidationFeedback(null)}
        />
      )}

      <div className="space-y-6">
        {voicesLoading ? (
          <div className="flex justify-center items-center py-12">
            <LoadingSpinner size="lg" />
          </div>
        ) : (
          <>
            {customVoices.length > 0 && (
              <div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">Custom Voices ({customVoices.length})</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {customVoices.map((voice) => (
                    <VoiceCard
                      key={voice.id}
                      voice={voice}
                      apiBaseUrl={settings.apiEndpoint}
                      onDelete={handleDeleteVoice}
                      onEdit={handleEditVoice}
                      onViewProfile={handleViewProfile}
                      isDeleting={deletingId === voice.id}
                      hasProfile={voiceProfiles[voice.id]}
                    />
                  ))}
                </div>
              </div>
            )}

            {defaultVoices.length > 0 && (
              <div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                  Default Voices ({defaultVoices.length})
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {defaultVoices.map((voice) => (
                    <VoiceCard key={voice.id} voice={voice} apiBaseUrl={settings.apiEndpoint} />
                  ))}
                </div>
              </div>
            )}

            {voices.length === 0 && (
              <div className="text-center py-16 px-4 border border-dashed border-gray-300 rounded-lg bg-gray-50">
                <p className="text-gray-600 font-medium">No voices available</p>
                <p className="mt-1 text-sm text-gray-500">
                  Create a custom voice from audio files, or ensure default voices are configured.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {editingId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md max-h-[90vh] flex flex-col">
            <h2 className="text-xl font-semibold text-gray-900 p-6 pb-0">Edit Voice</h2>
            <div className="p-6 overflow-y-auto flex-1 space-y-4">
              <Input
                label="Voice Name"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                required
                placeholder="e.g., My Custom Voice"
              />
              <Input
                label="Description (Optional)"
                multiline
                rows={3}
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                placeholder="Describe this voice..."
              />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Language (Optional)"
                  options={[{ value: '', label: 'Unknown' }, { value: 'in', label: 'Indian' }, ...SUPPORTED_LANGUAGES]}
                  value={editLanguageCode}
                  onChange={(e) => setEditLanguageCode(e.target.value)}
                />
                <Select
                  label="Gender (Optional)"
                  options={[
                    { value: 'unknown', label: 'Unknown' },
                    { value: 'female', label: 'Female' },
                    { value: 'male', label: 'Male' },
                    { value: 'neutral', label: 'Gender-neutral' },
                  ]}
                  value={editGender}
                  onChange={(e) => setEditGender(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Avatar image (optional)
                </label>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                  onChange={(e) => setEditImage(e.target.files?.[0] ?? null)}
                />
                <p className="mt-1 text-xs text-gray-500">JPEG, PNG or WebP. Max 5MB. Replaces current avatar.</p>
              </div>
            </div>
            <div className="flex gap-3 p-6 pt-4 border-t border-gray-200">
              <Button
                variant="primary"
                onClick={handleSaveEdit}
                isLoading={apiLoading}
                className="flex-1"
              >
                Save
              </Button>
              <Button
                variant="secondary"
                onClick={handleCancelEdit}
                disabled={apiLoading}
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}

      {profileModalVoiceId && (
        <VoiceProfileModal
          voiceId={profileModalVoiceId}
          voiceName={voices.find((v) => v.id === profileModalVoiceId)?.name || 'Unknown'}
          isOpen={!!profileModalVoiceId}
          onClose={() => setProfileModalVoiceId(null)}
          onGetProfile={getVoiceProfile}
          onUpdateProfile={handleUpdateProfileKeywords}
          onGenerateProfile={handleGenerateProfile}
        />
      )}

      {profileFromAudioOpen && (
        <VoiceProfileFromAudioModal
          isOpen={profileFromAudioOpen}
          onClose={() => setProfileFromAudioOpen(false)}
          voices={voices}
          defaultOllamaUrl={settings.ollamaServerUrl}
          defaultOllamaModel={settings.ollamaModel}
          onAnalyze={analyzeVoiceProfileFromAudio}
          onApply={handleApplyProfileFromAudio}
        />
      )}

      {createVoiceModalOpen && (
        <CreateVoiceModal
          isOpen={createVoiceModalOpen}
          onClose={() => setCreateVoiceModalOpen(false)}
          onCreate={handleCreateVoiceUnified}
        />
      )}
    </div>
  );
}