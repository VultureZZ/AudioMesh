import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert } from '../components/Alert';
import { Button } from '../components/Button';
import { Input } from '../components/Input';
import { Select } from '../components/Select';
import { useRealtimeSpeech } from '../hooks/useRealtimeSpeech';
import { useSettings } from '../hooks/useSettings';

const SAMPLE_RATE = 24000;

function pcm16ToFloat32(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  return out;
}

export function RealtimeGeneratePage() {
  const { settings } = useSettings();

  const [text, setText] = useState('Hello! This is AudioMesh realtime streaming audio.');
  const [voice, setVoice] = useState<string>('');
  const [upstreamVoices, setUpstreamVoices] = useState<string[]>([]);
  const [upstreamDefaultVoice, setUpstreamDefaultVoice] = useState<string | null>(null);
  const [cfgScale, setCfgScale] = useState<number>(1.5);
  const [inferenceSteps, setInferenceSteps] = useState<number>(5);
  const [volume, setVolume] = useState<number>(1.0);

  const { state, lastError, messages, audioStats, connect, disconnect, sendStart, sendText, sendFlush, sendStop, onAudioChunkRef } =
    useRealtimeSpeech(settings);

  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const nextPlayTimeRef = useRef<number>(0);

  const ensureAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
      nextPlayTimeRef.current = 0;
      gainNodeRef.current = audioContextRef.current.createGain();
      gainNodeRef.current.gain.value = volume;
      gainNodeRef.current.connect(audioContextRef.current.destination);
    }
    if (audioContextRef.current.state !== 'running') {
      await audioContextRef.current.resume();
    }
  }, []);

  const resetAudio = useCallback(async () => {
    const ctx = audioContextRef.current;
    audioContextRef.current = null;
    gainNodeRef.current = null;
    nextPlayTimeRef.current = 0;
    if (ctx) {
      try {
        await ctx.close();
      } catch {
        // ignore
      }
    }
  }, []);

  const playChunk = useCallback(
    async (chunk: ArrayBuffer) => {
      await ensureAudioContext();
      const ctx = audioContextRef.current;
      const gain = gainNodeRef.current;
      if (!ctx) return;

      const pcm = new Int16Array(chunk);
      const float = pcm16ToFloat32(pcm);

      const buffer = ctx.createBuffer(1, float.length, SAMPLE_RATE);
      // TS lib defs can be stricter about the underlying ArrayBuffer type;
      // copyToChannel only cares about the Float32Array view.
      buffer.copyToChannel(new Float32Array(float), 0);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gain ?? ctx.destination);

      const now = ctx.currentTime;
      const startAt = Math.max(nextPlayTimeRef.current, now + 0.05);
      source.start(startAt);
      nextPlayTimeRef.current = startAt + float.length / SAMPLE_RATE;
    },
    [ensureAudioContext]
  );

  // Keep gain in sync
  useEffect(() => {
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = volume;
    }
  }, [volume]);

  // Wire audio callback from the WS hook
  useEffect(() => {
    onAudioChunkRef.current = (chunk) => {
      void playChunk(chunk);
    };
    return () => {
      onAudioChunkRef.current = null;
    };
  }, [onAudioChunkRef, playChunk]);

  // Reset audio when disconnecting
  useEffect(() => {
    if (state === 'disconnected') {
      void resetAudio();
    }
  }, [resetAudio, state]);

  const canSend = state === 'connected';

  const statusLines = useMemo(() => {
    const filtered = messages
      .filter((m) => m && typeof m === 'object' && 'type' in m)
      .slice(-12);
    return filtered.map((m) => JSON.stringify(m));
  }, [messages]);

  // Extract upstream voice presets from status messages emitted by the backend.
  useEffect(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i] as any;
      if (m?.type === 'status' && m?.event === 'upstream_voice_presets' && m?.data) {
        const voices = Array.isArray(m.data.voices) ? (m.data.voices as string[]) : [];
        const def = typeof m.data.default_voice === 'string' ? (m.data.default_voice as string) : null;
        setUpstreamVoices(voices);
        setUpstreamDefaultVoice(def);
        // If no voice selected yet, set to default to avoid silent fallback surprises.
        if (!voice && def) setVoice(def);
        break;
      }
    }
  }, [messages, voice]);

  const handleConnect = async () => {
    connect();
    // Audio must be enabled by user gesture in most browsers, so also prepare it here.
    await ensureAudioContext();
  };

  const handleStart = () => {
    sendStart({
      cfg_scale: cfgScale,
      inference_steps: inferenceSteps,
      voice: voice.trim() || undefined,
    });
  };

  const handleSendAndFlush = () => {
    // Ensure the backend has the latest session parameters (voice/cfg/steps)
    // before starting generation.
    handleStart();
    sendText(text);
    sendFlush();
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Realtime Speech</h1>
        <p className="mt-2 text-gray-600">
          Stream PCM audio over WebSocket using VibeVoice-Realtime-0.5B.
        </p>
      </div>

      {lastError && <Alert type="error" message={lastError} />}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div className="flex flex-wrap gap-3">
          {state !== 'connected' ? (
            <Button variant="primary" onClick={handleConnect}>
              Connect
            </Button>
          ) : (
            <Button
              variant="secondary"
              onClick={() => {
                sendStop();
                disconnect();
              }}
            >
              Disconnect
            </Button>
          )}
          <div className="text-sm text-gray-600 flex items-center">
            Status: <span className="ml-2 font-medium text-gray-900">{state}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {upstreamVoices.length > 0 ? (
            <Select
              label="Voice preset (realtime model presets)"
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              options={[
                ...(upstreamDefaultVoice
                  ? [{ value: upstreamDefaultVoice, label: `${upstreamDefaultVoice} (default)` }]
                  : []),
                ...upstreamVoices
                  .filter((v) => v !== upstreamDefaultVoice)
                  .map((v) => ({ value: v, label: v })),
              ]}
            />
          ) : (
            <Input
              label="Voice preset (optional)"
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              placeholder="Will populate after first generation"
              helpText="Realtime voices come from the upstream demo server (GET /config). Custom voices from this app are not available in realtime mode."
            />
          )}
          <Input
            label="CFG scale"
            value={cfgScale.toString()}
            onChange={(e) => setCfgScale(parseFloat(e.target.value))}
            placeholder="1.5"
          />
          <Input
            label="Inference steps"
            value={inferenceSteps.toString()}
            onChange={(e) => setInferenceSteps(parseInt(e.target.value, 10))}
            placeholder="5"
          />
        </div>

        <div className="flex flex-wrap gap-3">
          <Button variant="secondary" onClick={handleStart} disabled={!canSend}>
            Start Session
          </Button>
          <Button
            variant="primary"
            onClick={handleSendAndFlush}
            disabled={!canSend || !text.trim()}
          >
            Send + Flush
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              sendFlush();
            }}
            disabled={!canSend}
          >
            Flush Only
          </Button>
          <Button
            variant="secondary"
            onClick={() => {
              sendStop();
              void resetAudio();
            }}
            disabled={!canSend}
          >
            Stop
          </Button>
        </div>

        <Input
          label="Text"
          multiline
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type text to synthesize..."
          required
        />

        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-900">Session log</div>
          <pre className="text-xs bg-gray-50 border rounded-md p-3 overflow-auto max-h-64">
{statusLines.join('\n')}
          </pre>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-xs text-gray-600">
              <div className="font-medium text-gray-900">Audio stats</div>
              <div>Chunks: {audioStats.chunks}</div>
              <div>Bytes: {audioStats.bytes}</div>
              <div>Seconds (est): {(audioStats.bytes / 2 / SAMPLE_RATE).toFixed(2)}</div>
            </div>
            <div className="text-xs text-gray-600">
              <div className="font-medium text-gray-900">Playback</div>
              <div>
                AudioContext:{' '}
                {audioContextRef.current
                  ? `${audioContextRef.current.state} @ ${audioContextRef.current.sampleRate}Hz`
                  : 'not created'}
              </div>
              <div>
                Queue lead:{' '}
                {audioContextRef.current
                  ? `${Math.max(0, nextPlayTimeRef.current - audioContextRef.current.currentTime).toFixed(2)}s`
                  : '0s'}
              </div>
            </div>
            <div className="text-xs text-gray-600">
              <div className="font-medium text-gray-900">Volume</div>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="w-full"
              />
              <div>{volume.toFixed(2)}x</div>
              <div className="mt-2 flex gap-2">
                <Button
                  variant="secondary"
                  onClick={() => {
                    void resetAudio();
                  }}
                >
                  Reset Audio
                </Button>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">
            Audio is streamed as raw PCM16LE mono @ 24000Hz and played via the Web Audio API.
          </div>
        </div>
      </div>
    </div>
  );
}

