"""
Sequential post-planning generation: ACE-Step for music, Stable Audio Open (optional) for SFX.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from app.services.asset_library import AssetLibrary
from app.services.prompt_router import PromptRouter
from app.services.stable_audio_client import stable_audio_open_client

if TYPE_CHECKING:
    from app.services.production_director import ProductionPlan

logger = logging.getLogger(__name__)


def _effective_acestep_duration_seconds(requested_seconds: float, category: str) -> float:
    """Clamp ACE-Step request duration: avoid tiny generations; cap at config max."""
    try:
        from vibevoice.config import config  # type: ignore

        lo = float(getattr(config, "ACESTEP_MIN_MUSIC_DURATION_SECONDS", 30.0))
        hi = float(getattr(config, "ACESTEP_MAX_MUSIC_DURATION_SECONDS", 600.0))
        if (category or "").strip() == "music_transition":
            lo = float(getattr(config, "ACESTEP_MIN_TRANSITION_DURATION_SECONDS", 10.0))
    except Exception:
        lo, hi = 30.0, 600.0
        if (category or "").strip() == "music_transition":
            lo = 10.0
    req = float(requested_seconds)
    out = max(lo, min(hi, req))
    if out > req + 1e-3:
        logger.info(
            "ACE-Step duration raised from %.2fs to %.2fs (category=%s, floor=%.2fs)",
            req,
            out,
            category,
            lo,
        )
    return out


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class GenerationJob:
    """One deferred generation (from Director tools and/or ProductionPlan asset refs)."""

    request_id: str
    category: str
    prompt: str
    duration_ms: int
    genre: str
    mood: str
    intensity: int = 3


class GenerationQueue:
    """
    Collects ``request_generation`` tool calls during planning, then after ``ProductionPlan``
    is finalized runs generations sequentially and patches ``generation_prompt`` refs to
    ``asset_id`` values. New assets are added to the library with ``source=ace_step_generated``.
    """

    def __init__(
        self,
        library: AssetLibrary,
        router: Optional[PromptRouter] = None,
        genre_template: Optional[Any] = None,
    ) -> None:
        self.library = library
        self.router = router or PromptRouter()
        self._genre_template = genre_template
        self._tool_jobs: List[GenerationJob] = []
        self.warnings: List[str] = []

    def enqueue_from_tool(
        self,
        *,
        category: str,
        prompt: str,
        duration_ms: int,
        genre: str,
        mood: str,
        intensity: int = 3,
    ) -> str:
        rid = f"req_{uuid.uuid4().hex[:12]}"
        self._tool_jobs.append(
            GenerationJob(
                request_id=rid,
                category=category,
                prompt=prompt.strip(),
                duration_ms=max(1, int(duration_ms)),
                genre=genre,
                mood=mood,
                intensity=max(1, min(5, int(intensity))),
            )
        )
        return rid

    def _collect_plan_prompts(self, plan: "ProductionPlan") -> List[GenerationJob]:
        jobs: List[GenerationJob] = []
        gtag = _normalize_tag(str(plan.genre)) if getattr(plan, "genre", None) else "general"
        for track in plan.tracks:
            if track.track_role in ("voice_main", "voice_backchannel"):
                continue
            for ev in track.events:
                ref = ev.asset_ref
                if not ref:
                    continue
                gp = getattr(ref, "generation_prompt", None)
                if not gp:
                    continue
                cat = _category_from_track_role(track.track_role)
                jobs.append(
                    GenerationJob(
                        request_id=f"plan_{uuid.uuid4().hex[:10]}",
                        category=cat,
                        prompt=str(gp).strip(),
                        duration_ms=max(1, int(ev.duration_ms)),
                        genre=gtag,
                        mood="neutral",
                        intensity=3,
                    )
                )
        return jobs

    async def run_queued_and_patch_plan(self, plan: "ProductionPlan") -> "ProductionPlan":
        """Generate assets for tool-queued jobs and plan ``generation_prompt`` fields; patch plan."""
        from app.services.production_director import AssetRef, ProductionPlan

        plan_jobs = self._collect_plan_prompts(plan)
        ordered: List[GenerationJob] = list(self._tool_jobs)
        seen_prompts: Set[str] = {j.prompt for j in self._tool_jobs}
        for j in plan_jobs:
            if j.prompt not in seen_prompts:
                ordered.append(j)
                seen_prompts.add(j.prompt)

        prompt_to_asset: Dict[str, str] = {}
        for job in ordered:
            if job.prompt in prompt_to_asset:
                continue
            try:
                aid = await self._generate_one(job)
                if aid:
                    prompt_to_asset[job.prompt] = aid
            except Exception as exc:
                w = f"Generation failed for prompt ({job.category}): {exc}"
                logger.warning(w)
                self.warnings.append(w)

        return self._patch_plan(plan, prompt_to_asset, ProductionPlan, AssetRef)

    def _patch_plan(
        self,
        plan: "ProductionPlan",
        mapping: Dict[str, str],
        ProductionPlanCls: Any,
        AssetRefCls: Any,
    ) -> "ProductionPlan":
        raw = plan.model_dump(mode="json")
        for t in raw.get("tracks") or []:
            for e in t.get("events") or []:
                ar = e.get("asset_ref")
                if not ar or not isinstance(ar, dict):
                    continue
                gp = ar.get("generation_prompt")
                if not gp:
                    continue
                aid = mapping.get(str(gp).strip())
                if aid:
                    e["asset_ref"] = {"asset_id": aid}
                    try:
                        asset = self.library.get(aid)
                        e["duration_ms"] = int(asset.duration_ms)
                    except Exception as exc:
                        logger.debug("Could not sync event duration from asset %s: %s", aid, exc)
        return ProductionPlanCls.model_validate(raw)

    async def generate_for_track_event(
        self,
        *,
        track_role: str,
        prompt: str,
        duration_ms: int,
        plan_genre: str,
        mood: str = "neutral",
        intensity: int = 3,
    ) -> Optional[str]:
        """Regenerate a single stem (ACE-Step / Stable Audio) for operator audition workflows."""
        cat = _category_from_track_role(track_role)
        job = GenerationJob(
            request_id=f"regen_{uuid.uuid4().hex[:12]}",
            category=cat,
            prompt=prompt.strip(),
            duration_ms=max(1, int(duration_ms)),
            genre=_normalize_tag(str(plan_genre)),
            mood=mood,
            intensity=max(1, min(5, int(intensity))),
        )
        return await self._generate_one(job)

    async def _generate_one(self, job: GenerationJob) -> Optional[str]:
        backend = self.router.route(job.category)
        if backend == "skip":
            self.warnings.append(f"Skipped generation (backend=skip) for category={job.category}")
            return None
        if backend == "acestep":
            return await self._generate_acestep(job)
        if backend == "stable_audio":
            return await self._generate_stable_audio(job)
        return None

    async def _generate_acestep(self, job: GenerationJob) -> str:
        from vibevoice.services.music_generator import music_generator

        prompt = self.router.apply_genre_prompt_modifiers(
            job.prompt, job.category, self._genre_template
        )
        duration_s = _effective_acestep_duration_seconds(job.duration_ms / 1000.0, job.category)
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "instrumental": True,
            "thinking": True,
            "batch_size": 1,
            "inference_steps": 8,
            "seed": -1,
            "audio_format": "wav",
            "duration": duration_s,
        }
        task_id = await music_generator.generate_music(payload)
        st = await self._poll_task(music_generator, task_id)
        meta_list = st.get("metadata") or []
        if not meta_list:
            raise RuntimeError("ACE-Step returned no audio metadata")
        src = Path(meta_list[0].get("file_path") or "")
        if not src.is_file():
            raise RuntimeError(f"ACE-Step output missing: {src}")

        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / "out.wav"
            await asyncio.to_thread(_copy_to_wav, src, tmp_wav)
            duration_ms = await asyncio.to_thread(_wav_duration_ms, tmp_wav)
            meta = {
                "category": job.category,
                "genre_tags": [_normalize_tag(job.genre)],
                "mood_tags": [_normalize_tag(job.mood)],
                "intensity": job.intensity,
                "source": "ace_step_generated",
                "licensing": "ACE-Step generated",
                "duration_ms": duration_ms,
            }
            return self.library.add_asset(tmp_wav, meta)

    async def _poll_task(self, music_generator: Any, task_id: str, poll_s: float = 3.0) -> Dict[str, Any]:
        while True:
            st = await music_generator.get_status(task_id)
            status = st.get("status")
            if status == "succeeded":
                return st
            if status == "failed":
                raise RuntimeError(st.get("error") or "ACE-Step failed")
            await asyncio.sleep(poll_s)

    async def _generate_stable_audio(self, job: GenerationJob) -> Optional[str]:
        prompt = self.router.apply_genre_prompt_modifiers(
            job.prompt, job.category, self._genre_template
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sfx.wav"
            duration_s = max(0.5, min(120.0, job.duration_ms / 1000.0))
            result = await stable_audio_open_client.generate(
                prompt=prompt,
                duration_seconds=duration_s,
                out_path=out,
            )
            if result is None or not result.is_file():
                self.warnings.append(
                    f"Stable Audio Open not available; skipped SFX: {job.prompt[:80]}"
                )
                return None
            duration_ms = await asyncio.to_thread(_wav_duration_ms, result)
            meta = {
                "category": job.category,
                "genre_tags": [_normalize_tag(job.genre)],
                "mood_tags": [_normalize_tag(job.mood)],
                "intensity": job.intensity,
                "source": "user_uploaded",
                "licensing": "stable audio open (pending integration)",
                "duration_ms": duration_ms,
            }
            return self.library.add_asset(result, meta)


def _category_from_track_role(track_role: str) -> str:
    """Map ProductionPlan track_role to AssetLibrary category for generation."""
    role = (track_role or "").strip()
    mapping = {
        "music_bed": "music_bed",
        "music_transition": "music_transition",
        "music_intro": "music_intro",
        "music_outro": "music_outro",
        "sfx_impact": "sfx_impact",
        "sfx_riser": "sfx_riser",
        "sfx_whoosh": "sfx_whoosh",
        "sfx_ambience": "sfx_ambience",
        "sfx_laugh": "sfx_laugh",
        "sfx_reveal": "sfx_reveal",
        "foley": "foley",
    }
    return mapping.get(role, "music_bed")


def _copy_to_wav(src: Path, dest: Path) -> None:
    import shutil

    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dest)
        return
    try:
        import librosa

        y, sr = librosa.load(str(src), sr=None, mono=True)
        sf.write(str(dest), y, int(sr))
    except Exception:
        from pydub import AudioSegment

        AudioSegment.from_file(str(src)).export(str(dest), format="wav")


def _wav_duration_ms(path: Path) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return int(round(float(info.duration) * 1000.0))

