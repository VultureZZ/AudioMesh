import { Button } from '../Button';

interface ExportControlsProps {
  onDownloadPdf: () => Promise<void>;
  onDownloadJson: () => Promise<void>;
  onCopyTranscript: () => void;
  onGenerateFromTranscript?: () => void;
  canGenerateFromTranscript?: boolean;
  isLoading?: boolean;
}

export function ExportControls({
  onDownloadPdf,
  onDownloadJson,
  onCopyTranscript,
  onGenerateFromTranscript,
  canGenerateFromTranscript = false,
  isLoading = false,
}: ExportControlsProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-3">
      <h3 className="text-lg font-semibold text-gray-900">Export</h3>
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" onClick={onDownloadPdf} isLoading={isLoading}>
          Download PDF
        </Button>
        <Button variant="secondary" onClick={onDownloadJson} isLoading={isLoading}>
          Download JSON
        </Button>
        <Button variant="secondary" onClick={onCopyTranscript}>
          Copy Transcript
        </Button>
        {canGenerateFromTranscript && onGenerateFromTranscript && (
          <Button variant="primary" onClick={onGenerateFromTranscript}>
            Generate from transcript
          </Button>
        )}
      </div>
    </div>
  );
}

