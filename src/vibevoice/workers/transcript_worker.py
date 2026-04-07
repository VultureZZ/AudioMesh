"""
External transcript worker entrypoint.

Run with a transcript-dedicated Python environment, for example:
  .venv-transcripts/bin/python src/vibevoice/workers/transcript_worker.py process <id> <wav_path>
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    # Ensure `src/` is importable when invoked as a plain script.
    this_file = Path(__file__).resolve()
    src_dir = this_file.parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


async def _main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: transcript_worker.py <process|analyze> <transcript_id> [wav_path]")
        return 2

    action = argv[1].strip().lower()
    transcript_id = argv[2].strip()
    wav_path = argv[3].strip() if len(argv) > 3 else None

    from vibevoice.core.transcripts.pipeline import transcript_pipeline

    if action == "process":
        if not wav_path:
            print("Missing wav_path for process action")
            return 2
        await transcript_pipeline.process_transcript(transcript_id, wav_path)
        return 0

    if action == "analyze":
        await transcript_pipeline.run_analysis(transcript_id)
        return 0

    print(f"Unknown action: {action}")
    return 2


if __name__ == "__main__":
    _bootstrap_path()
    raise SystemExit(asyncio.run(_main(sys.argv)))

