"""
Transcript report generation (PDF/JSON/Markdown).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ...config import config


class TranscriptReporter:
    async def generate_pdf(
        self,
        transcript_id: str,
        analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        *,
        title: str,
        recording_type: str,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_pdf_sync,
            transcript_id,
            analysis,
            segments,
            speakers,
            title,
            recording_type,
        )

    def _generate_pdf_sync(
        self,
        transcript_id: str,
        analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        title: str,
        recording_type: str,
    ) -> str:
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas
        except Exception as exc:
            raise RuntimeError(
                "PDF report generation requires reportlab. Install with: pip install reportlab"
            ) from exc

        reports_dir = config.TRANSCRIPTS_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{transcript_id}.pdf"

        c = canvas.Canvas(str(path), pagesize=LETTER)
        width, height = LETTER
        y = height - 60

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, f"{title} ({recording_type.title()})")
        y -= 28

        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Transcript ID: {transcript_id}")
        y -= 18
        c.drawString(50, y, f"Speakers: {', '.join([s.get('label') or s.get('id') for s in speakers])}")
        y -= 24

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Summary")
        y -= 16
        c.setFont("Helvetica", 10)
        for line in self._wrap(analysis.get("summary", ""), 95):
            c.drawString(50, y, line)
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 60

        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Action Items")
        y -= 16
        c.setFont("Helvetica", 10)
        for item in analysis.get("action_items", []):
            row = f"- {item.get('action')} (owner: {item.get('owner') or 'n/a'}, priority: {item.get('priority', 'medium')})"
            for line in self._wrap(row, 95):
                c.drawString(50, y, line)
                y -= 14
                if y < 80:
                    c.showPage()
                    y = height - 60

        c.showPage()
        c.save()
        return str(path)

    async def generate_json(
        self,
        transcript_id: str,
        analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        *,
        title: str,
        recording_type: str,
    ) -> str:
        reports_dir = config.TRANSCRIPTS_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{transcript_id}.json"
        payload = {
            "transcript_id": transcript_id,
            "title": title,
            "recording_type": recording_type,
            "analysis": analysis,
            "speakers": speakers,
            "transcript": segments,
        }
        await asyncio.to_thread(path.write_text, json.dumps(payload, indent=2))
        return str(path)

    async def generate_markdown(
        self,
        transcript_id: str,
        analysis: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: list[dict[str, Any]],
        *,
        title: str,
        recording_type: str,
    ) -> str:
        reports_dir = config.TRANSCRIPTS_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{transcript_id}.md"

        lines = [
            f"# {title}",
            "",
            f"- Recording type: `{recording_type}`",
            f"- Transcript ID: `{transcript_id}`",
            f"- Speakers: {', '.join([s.get('label') or s.get('id') for s in speakers])}",
            "",
            "## Summary",
            analysis.get("summary", ""),
            "",
            "## Action Items",
        ]
        for item in analysis.get("action_items", []):
            lines.append(
                f"- {item.get('action')} (owner: {item.get('owner') or 'n/a'}, due: {item.get('due_hint') or 'n/a'}, priority: {item.get('priority', 'medium')})"
            )
        lines.extend(["", "## Key Decisions"])
        for x in analysis.get("key_decisions", []):
            lines.append(f"- {x}")
        lines.extend(["", "## Open Questions"])
        for x in analysis.get("open_questions", []):
            lines.append(f"- {x}")

        await asyncio.to_thread(path.write_text, "\n".join(lines))
        return str(path)

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        words = (text or "").split()
        out: list[str] = []
        line: list[str] = []
        count = 0
        for w in words:
            if count + len(w) + (1 if line else 0) > width:
                out.append(" ".join(line))
                line = [w]
                count = len(w)
            else:
                line.append(w)
                count += len(w) + (1 if len(line) > 1 else 0)
        if line:
            out.append(" ".join(line))
        return out


transcript_reporter = TranscriptReporter()

