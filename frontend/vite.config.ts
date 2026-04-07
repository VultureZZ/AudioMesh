import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Allow both local dev and systemd service usage without hardcoding a domain.
  // Vite will only expose env vars prefixed with VITE_ to the browser; here we're
  // using loadEnv for config-time values (server/preview/proxy).
  const env = loadEnv(mode, process.cwd(), '');

  // Prefer .env values (loadEnv), but allow systemd EnvironmentFile to control
  // config-time behavior too.
  const get = (key: string): string | undefined => env[key] ?? process.env[key];

  const host = get('VITE_DEV_HOST') || '0.0.0.0';
  const port = Number(get('VITE_DEV_PORT') || '3001');
  const apiProxyTarget = get('VITE_API_PROXY_TARGET') || 'http://localhost:8000';

  // Fix: when running as a service behind a real hostname / reverse proxy,
  // Vite's host check can block requests unless the host is explicitly allowed.
  //
  // Set e.g.:
  //   VITE_ALLOWED_HOSTS=your-hostname.example.com
  // or (less strict):
  //   VITE_ALLOWED_HOSTS=all
  const allowedHostsRaw = (get('VITE_ALLOWED_HOSTS') || '').trim();
  const allowedHosts =
    allowedHostsRaw.toLowerCase() === 'all'
      ? true
      : allowedHostsRaw
        ? allowedHostsRaw
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : host === '127.0.0.1' || host === 'localhost'
          ? undefined
          : true;

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      host,
      port,
      ...(allowedHosts !== undefined ? { allowedHosts } : {}),
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      host,
      port,
      ...(allowedHosts !== undefined ? { allowedHosts } : {}),
    },
  };
});
