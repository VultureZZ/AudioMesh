/**
 * Custom hook for voice management with caching
 */

import { useState, useEffect, useCallback } from 'react';
import { VoiceResponse } from '../types/api';
import { useApi } from './useApi';
import { getVoiceDisplayName } from '../utils/format';

export function useVoices() {
  const { listVoices: apiListVoices, loading, error } = useApi();
  const [voices, setVoices] = useState<VoiceResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sortVoicesAlpha = (items: VoiceResponse[]): VoiceResponse[] => {
    return [...items].sort((a, b) => {
      const aKey = getVoiceDisplayName(a).toLocaleLowerCase();
      const bKey = getVoiceDisplayName(b).toLocaleLowerCase();
      return aKey.localeCompare(bKey);
    });
  };

  const fetchVoices = useCallback(async () => {
    setIsLoading(true);
    const response = await apiListVoices();
    if (response) {
      setVoices(sortVoicesAlpha(response.voices));
    }
    setIsLoading(false);
  }, [apiListVoices]);

  useEffect(() => {
    fetchVoices();
  }, [fetchVoices]);

  return {
    voices,
    loading: isLoading || loading,
    error,
    refresh: fetchVoices,
  };
}