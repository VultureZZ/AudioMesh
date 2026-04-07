/**
 * Validation utility functions
 */

/**
 * Validate API endpoint URL
 */
export function validateApiEndpoint(url: string): { valid: boolean; error?: string } {
  if (!url.trim()) {
    return { valid: false, error: 'API endpoint is required' };
  }

  try {
    const parsedUrl = new URL(url);
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      return { valid: false, error: 'API endpoint must use HTTP or HTTPS' };
    }
    return { valid: true };
  } catch {
    return { valid: false, error: 'Invalid URL format' };
  }
}

/**
 * Validate audio file type
 */
export function isValidAudioFile(file: File): boolean {
  const validTypes = [
    'audio/wav',
    'audio/wave',
    'audio/x-wav',
    'audio/mpeg',
    'audio/mp3',
    'audio/mp4',
    'audio/x-m4a',
    'audio/ogg',
    'audio/webm',
    'audio/flac',
  ];

  const validExtensions = ['.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac', '.mp4'];
  const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

  return (
    validTypes.includes(file.type) ||
    validExtensions.includes(fileExtension) ||
    file.type.startsWith('audio/')
  );
}

/**
 * Validate file size (in bytes)
 */
export function validateFileSize(file: File, maxSizeMB: number = 100): { valid: boolean; error?: string } {
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxSizeBytes) {
    return {
      valid: false,
      error: `File size exceeds ${maxSizeMB}MB limit (${(file.size / 1024 / 1024).toFixed(2)}MB)`,
    };
  }
  return { valid: true };
}

/**
 * Validate voice name
 */
export function validateVoiceName(name: string): { valid: boolean; error?: string } {
  if (!name.trim()) {
    return { valid: false, error: 'Voice name is required' };
  }
  if (name.length < 1) {
    return { valid: false, error: 'Voice name must be at least 1 character' };
  }
  if (name.length > 100) {
    return { valid: false, error: 'Voice name must be less than 100 characters' };
  }
  // Check for invalid characters
  if (!/^[a-zA-Z0-9\s\-_]+$/.test(name)) {
    return {
      valid: false,
      error: 'Voice name can only contain letters, numbers, spaces, hyphens, and underscores',
    };
  }
  return { valid: true };
}