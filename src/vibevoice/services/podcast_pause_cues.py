"""
Contextual [PAUSE_MS:N] values for multi-speaker podcast TTS.

Keeps variation deterministic (same script yields same timings) without a flat default on every line.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .tts.segments import INLINE_PAUSE_MS_PATTERN

PAUSE_MS_MIN = 120
PAUSE_MS_MAX = 480

_SPEAKER_HEAD = re.compile(r"^(Speaker\s+(\d+)\s*:\s*)(.*)$", re.IGNORECASE | re.DOTALL)


def contextual_pause_ms_for_handoff(
    spoken_body: str,
    *,
    handoff_index: int,
    from_speaker: int,
    to_speaker: int,
) -> int:
    """
    Milliseconds of silence after this speaker line before the next voice, tuned to text and position.

    ``spoken_body`` must not include ``[PAUSE_MS:...]`` tokens.
    """
    text = (spoken_body or "").strip()
    if not text:
        return max(PAUSE_MS_MIN, min(PAUSE_MS_MAX, 200 + (handoff_index % 7) * 9))

    wc = len(text.split())
    t = text.rstrip()
    score = 178

    if t.endswith("?"):
        score += 95
    elif t.endswith("!"):
        score += 58
    elif t.endswith("..."):
        score += 42
    if "..." in text[-24:]:
        score += 28
    if "—" in text or " – " in text:
        score += 18

    if wc <= 10:
        score -= 48
    elif wc <= 22:
        score -= 18
    elif wc >= 52:
        score += 52
    elif wc >= 38:
        score += 28

    if from_speaker != to_speaker:
        score += 12

    ripple = (handoff_index * 41 + from_speaker * 19 + to_speaker * 29) % 113
    score += ripple - 56

    fp = sum(ord(c) for c in text[:80]) % 67
    score += fp - 33

    return max(PAUSE_MS_MIN, min(PAUSE_MS_MAX, score))


def redistribute_uniform_pause_markers(script: str) -> str:
    """
    If almost every speaker line ends with the same ``[PAUSE_MS:N]`` (typical lazy copy-paste),
    replace each with a contextual value so gaps feel human and varied.

    When the model already uses a spread of values, the script is unchanged.
    """
    lines = script.split("\n")
    speaker_meta: List[Tuple[int, int, str, Optional[int]]] = []
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        m = _SPEAKER_HEAD.match(s)
        if not m:
            continue
        sn = int(m.group(2))
        rest = m.group(3).strip()
        last_ms: Optional[int] = None
        for pm in INLINE_PAUSE_MS_PATTERN.finditer(rest):
            try:
                last_ms = int(pm.group(1))
            except ValueError:
                last_ms = None
        spoken = INLINE_PAUSE_MS_PATTERN.sub(" ", rest)
        spoken = re.sub(r" +", " ", spoken).strip()
        speaker_meta.append((idx, sn, spoken, last_ms))

    if len(speaker_meta) < 3:
        return script

    ms_values = [m[3] for m in speaker_meta if m[3] is not None]
    if len(ms_values) < 3:
        return script

    # All present markers identical → model copy-pasted one number (e.g. 220 everywhere).
    if len(set(ms_values)) != 1:
        return script

    # Next speaker for each row (skip non-speaker lines in file by walking speaker_meta only)
    out_lines = list(lines)
    for k, (line_idx, from_sn, spoken, _old_ms) in enumerate(speaker_meta):
        to_sn: Optional[int] = None
        if k + 1 < len(speaker_meta):
            to_sn = speaker_meta[k + 1][1]
        else:
            to_sn = from_sn
        new_ms = contextual_pause_ms_for_handoff(
            spoken,
            handoff_index=k,
            from_speaker=from_sn,
            to_speaker=to_sn or from_sn,
        )
        raw = lines[line_idx].strip()
        m = _SPEAKER_HEAD.match(raw)
        if not m:
            continue
        prefix = m.group(1)
        rest = m.group(3).strip()
        rest_wo = INLINE_PAUSE_MS_PATTERN.sub(" ", rest)
        rest_wo = re.sub(r" +", " ", rest_wo).strip()
        out_lines[line_idx] = f"{prefix}{rest_wo} [PAUSE_MS:{new_ms}]"

    return "\n".join(out_lines)
