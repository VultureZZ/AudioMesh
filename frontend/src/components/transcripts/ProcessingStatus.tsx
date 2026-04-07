import type { TranscriptStatusResponse } from '../../types/api';

interface ProcessingStatusProps {
  status: TranscriptStatusResponse;
}

export function ProcessingStatus({ status }: ProcessingStatusProps) {
  const pct = Math.max(0, Math.min(100, status.progress_pct || 0));
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-3">
      <h3 className="text-lg font-semibold text-gray-900">Processing</h3>
      <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full bg-primary-600 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="flex justify-between text-sm text-gray-600">
        <span>{status.current_stage || status.status}</span>
        <span>{pct}%</span>
      </div>
      {status.error && <div className="text-sm text-red-600">{status.error}</div>}
    </div>
  );
}

