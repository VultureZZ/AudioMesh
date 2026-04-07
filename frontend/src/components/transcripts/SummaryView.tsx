import type { TranscriptAnalysis } from '../../types/api';

interface SummaryViewProps {
  analysis?: TranscriptAnalysis | null;
  recordingType: string;
}

export function SummaryView({ analysis, recordingType }: SummaryViewProps) {
  if (!analysis) return null;
  const showActionItems = recordingType !== 'memo';

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Summary</h3>
      <p className="text-gray-800 whitespace-pre-wrap">{analysis.summary}</p>

      {analysis.topics_discussed?.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Topics</h4>
          <div className="flex flex-wrap gap-2">
            {analysis.topics_discussed.map((topic, idx) => (
              <span key={`${topic}-${idx}`} className="px-2 py-1 bg-gray-100 rounded text-sm text-gray-700">
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}

      {showActionItems && analysis.action_items?.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Action items</h4>
          <ul className="list-disc ml-5 space-y-1 text-sm text-gray-800">
            {analysis.action_items.map((item, idx) => (
              <li key={`${item.action}-${idx}`}>
                {item.action} {item.owner ? `(owner: ${item.owner})` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      {analysis.key_decisions?.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Key decisions</h4>
          <ul className="list-disc ml-5 space-y-1 text-sm text-gray-800">
            {analysis.key_decisions.map((d, idx) => (
              <li key={`${d}-${idx}`}>{d}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

