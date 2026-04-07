import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppSettings } from '../types/settings';

type StatusMessage =
  | { type: 'status'; event: string; data?: unknown }
  | { type: 'error'; message: string }
  | { type: 'end' }
  | { type: string; [key: string]: unknown };

type RealtimeClientState = 'disconnected' | 'connecting' | 'connected';

function toWebSocketUrl(apiBaseUrl: string, path: string, apiKey?: string): string {
  const base = new URL(apiBaseUrl);
  const wsProtocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = new URL(base.toString());
  wsUrl.protocol = wsProtocol;
  wsUrl.pathname = path;
  if (apiKey) wsUrl.searchParams.set('api_key', apiKey);
  return wsUrl.toString();
}

export function useRealtimeSpeech(settings: AppSettings) {
  const [state, setState] = useState<RealtimeClientState>('disconnected');
  const [lastError, setLastError] = useState<string | null>(null);
  const [messages, setMessages] = useState<StatusMessage[]>([]);
  const [audioStats, setAudioStats] = useState<{ chunks: number; bytes: number }>({
    chunks: 0,
    bytes: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const onAudioChunkRef = useRef<((chunk: ArrayBuffer) => void) | null>(null);

  const wsUrl = useMemo(
    () => toWebSocketUrl(settings.apiEndpoint, '/api/v1/speech/realtime', settings.apiKey),
    [settings.apiEndpoint, settings.apiKey]
  );

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setLastError(null);
    setMessages([]);
    setAudioStats({ chunks: 0, bytes: 0 });
    setState('connecting');

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setState('connected');
    };

    ws.onclose = (evt) => {
      const reason = evt.reason ? ` (${evt.reason})` : '';
      // If we never reached OPEN, treat as handshake-ish failure.
      if (state === 'connecting') {
        setLastError(`WebSocket closed during connect: code=${evt.code}${reason}`);
      } else if (evt.code !== 1000) {
        setLastError(`WebSocket closed: code=${evt.code}${reason}`);
      }
      setState('disconnected');
      wsRef.current = null;
    };

    ws.onerror = () => {
      // Browser doesn't give much detail here; keep it but also log close codes in onclose.
      setLastError('WebSocket error (see server logs for details)');
    };

    ws.onmessage = async (evt) => {
      if (typeof evt.data === 'string') {
        try {
          const parsed = JSON.parse(evt.data) as StatusMessage;
          setMessages((prev) => [...prev, parsed].slice(-200));
          if (parsed.type === 'error' && typeof (parsed as any).message === 'string') {
            setLastError((parsed as any).message);
          }
        } catch {
          setMessages((prev) => [...prev, { type: 'status', event: 'text', data: evt.data }].slice(-200));
        }
        return;
      }

      // Binary audio chunk (PCM16 bytes).
      // Some browsers may deliver Blob even when binaryType is set.
      const chunk =
        evt.data instanceof Blob ? await evt.data.arrayBuffer() : (evt.data as ArrayBuffer);
      onAudioChunkRef.current?.(chunk);
      setAudioStats((prev) => ({ chunks: prev.chunks + 1, bytes: prev.bytes + chunk.byteLength }));
      setMessages((prev) =>
        [...prev, { type: 'status', event: 'audio_chunk', data: { bytes: chunk.byteLength } }].slice(-200)
      );
    };

    wsRef.current = ws;
  }, [wsUrl]);

  const disconnect = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
    }
    setState('disconnected');
  }, []);

  const send = useCallback((payload: unknown) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setLastError('Not connected');
      return;
    }
    ws.send(JSON.stringify(payload));
  }, []);

  const sendStart = useCallback(
    (params: { cfg_scale?: number; inference_steps?: number; voice?: string }) => {
      send({ type: 'start', ...params });
    },
    [send]
  );

  const sendText = useCallback(
    (text: string) => {
      send({ type: 'text', text });
    },
    [send]
  );

  const sendFlush = useCallback(() => {
    send({ type: 'flush' });
  }, [send]);

  const sendStop = useCallback(() => {
    send({ type: 'stop' });
  }, [send]);

  // Cleanup on unmount
  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return {
    state,
    lastError,
    messages,
    audioStats,
    connect,
    disconnect,
    sendStart,
    sendText,
    sendFlush,
    sendStop,
    ws: wsRef,
    onAudioChunkRef,
  };
}

