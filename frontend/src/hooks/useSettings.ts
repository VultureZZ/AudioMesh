/**
 * Custom hook for settings management
 */

import { useState, useEffect, useCallback } from 'react';
import { AppSettings, DEFAULT_SETTINGS } from '../types/settings';
import { storage } from '../services/storage';
import { apiClient } from '../services/api';

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [isLoading, setIsLoading] = useState(true);

  // Load settings on mount
  useEffect(() => {
    const loaded = storage.loadSettings();
    setSettings(loaded);
    apiClient.updateConfig(loaded);
    setIsLoading(false);
  }, []);

  // Save settings and update API client
  const saveSettings = useCallback((newSettings: AppSettings) => {
    storage.saveSettings(newSettings);
    setSettings(newSettings);
    apiClient.updateConfig(newSettings);
  }, []);

  // Clear settings
  const clearSettings = useCallback(() => {
    storage.clearSettings();
    const defaults = { ...DEFAULT_SETTINGS };
    setSettings(defaults);
    apiClient.updateConfig(defaults);
  }, []);

  return {
    settings,
    isLoading,
    saveSettings,
    clearSettings,
  };
}