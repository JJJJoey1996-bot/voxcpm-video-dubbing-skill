from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from .runtime import EngineConfig, GenerationRequest, VoxCPMEngine


DEFAULT_DEMUCS_MODEL = "mdx_q"
DEFAULT_MAX_SEGMENT_MS = 4500
DEFAULT_REFERENCE_TARGET_MS = 10000
DEFAULT_GLOBAL_REFERENCE_TARGET_MS = 20000
DEFAULT_MAX_ALLOWED_SPEEDUP = 1.10
DEFAULT_CHUNK_MERGE_GAP_MS = 350
DEFAULT_MAX_CHUNK_DURATION_MS = 18000
DEFAULT_MIN_PREDICTED_RATIO = 0.90
DEFAULT_TARGET_LANGUAGE = "zh"
PHONEME_RATIO_CALIBRATION = {
    "zh": 3.2,
    "yue": 3.2,
    "ja": 2.2,
}


@dataclass
class TranscriptSegment:
    index: int
    start_ms: int
    end_ms: int
    source_text: str
    duration_ms: int
    phoneme_budget: int
    translated_text: str = ""
    raw_generated_duration_ms: int = 0
    rendered_duration_ms: int = 0
    adjusted_duration_ms: int = 0
    timeline_start_ms: int = 0
    timeline_end_ms: int = 0
    translated_phoneme_budget: int = 0


@dataclass
class DubChunk:
    index: int
    start_ms: int
    end_ms: int
    source_text: str
    duration_ms: int
    phoneme_budget: int
    segment_indexes: list[int]
    translated_text: str = ""
    raw_generated_duration_ms: int = 0
    rendered_duration_ms: int = 0
    adjusted_duration_ms: int = 0
    timeline_start_ms: int = 0
    timeline_end_ms: int = 0
    translated_phoneme_budget: int = 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _calibration_store_path() -> Path:
    return _project_root() / "outputs" / "video_dub_calibration.json"


def _default_calibration_store() -> dict[str, Any]:
    return {
        "languages": {
            lang: {
                "phoneme_factor": factor,
                "min_ratio": DEFAULT_MIN_PREDICTED_RATIO,
                "samples": 0,
            }
            for lang, factor in PHONEME_RATIO_CALIBRATION.items()
        }
    }


def _load_calibration_store() -> dict[str, Any]:
    path = _calibration_store_path()
    if not path.exists():
        return _default_calibration_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_calibration_store()
    if not isinstance(data, dict):
        return _default_calibration_store()
    languages = data.setdefault("languages", {})
    for lang, factor in PHONEME_RATIO_CALIBRATION.items():
        entry = languages.setdefault(lang, {})
        entry.setdefault("phoneme_factor", factor)
        entry.setdefault("min_ratio", DEFAULT_MIN_PREDICTED_RATIO)
        entry.setdefault("samples", 0)
    return data


def _save_calibration_store(store: dict[str, Any]) -> None:
    path = _calibration_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_paths_under_outputs() -> list[Path]:
    return sorted((_project_root() / "outputs").glob("*/manifest.json"))


def _normalized_lang_prefix(lang: str) -> str:
    lowered = (lang or "").lower()
    for prefix in PHONEME_RATIO_CALIBRATION:
        if lowered.startswith(prefix):
            return prefix
    return lowered or DEFAULT_TARGET_LANGUAGE


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def update_calibration_from_manifests(
    target_lang: str = DEFAULT_TARGET_LANGUAGE,
    manifest_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    lang_prefix = _normalized_lang_prefix(target_lang)
    manifests = manifest_paths or _manifest_paths_under_outputs()
    factor_samples: list[float] = []
    actual_ratio_samples: list[float] = []

    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for chunk in manifest.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
            source_budget = int(chunk.get("phoneme_budget") or 0)
            translated_text = (chunk.get("translated_text") or "").strip()
            raw_generated_duration_ms = int(chunk.get("raw_generated_duration_ms") or 0)
            duration_ms = int(chunk.get("duration_ms") or 0)
            if source_budget <= 0 or raw_generated_duration_ms <= 0 or duration_ms <= 0 or not translated_text:
                continue
            translated_budget = int(chunk.get("translated_phoneme_budget") or 0) or _phoneme_budget(translated_text, target_lang)
            if translated_budget <= 0:
                continue
            actual_ratio = float(raw_generated_duration_ms) / float(duration_ms)
            factor = actual_ratio * float(source_budget) / float(translated_budget)
            if 1.5 <= factor <= 6.0:
                factor_samples.append(factor)
                actual_ratio_samples.append(actual_ratio)

    store = _load_calibration_store()
    lang_entry = store.setdefault("languages", {}).setdefault(lang_prefix, {})
    if factor_samples:
        tuned_factor = min(4.8, max(2.0, _median(factor_samples)))
        tuned_min_ratio = min(0.96, max(0.92, _median(actual_ratio_samples) - 0.03))
        lang_entry["phoneme_factor"] = round(tuned_factor, 3)
        lang_entry["min_ratio"] = round(tuned_min_ratio, 3)
        lang_entry["samples"] = len(factor_samples)
        _save_calibration_store(store)
    else:
        lang_entry.setdefault("phoneme_factor", PHONEME_RATIO_CALIBRATION.get(lang_prefix, 1.0))
        lang_entry.setdefault("min_ratio", DEFAULT_MIN_PREDICTED_RATIO)
        lang_entry.setdefault("samples", 0)
    return lang_entry


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=merged_env)


def _run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary not found: {name}")
    return path


def _ffprobe_duration(path: Path) -> float:
    _require_binary("ffprobe")
    out = _run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out)


def _extract_audio(video_path: Path, out_wav: Path) -> None:
    _require_binary("ffmpeg")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(out_wav),
        ]
    )


def _resample_for_whisper(input_wav: Path, output_wav: Path) -> None:
    _require_binary("ffmpeg")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_wav),
        ]
    )


def _separate_audio(audio_path: Path, out_dir: Path, model: str, device: str) -> tuple[Path, Path]:
    project_cache = Path.cwd() / ".cache" / "demucs"
    cache_root = project_cache if project_cache.exists() else out_dir.parent / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            model,
            "-d",
            device,
            "-o",
            str(out_dir),
            str(audio_path),
        ],
        env={
            "TORCH_HOME": str(cache_root / "torch"),
            "XDG_CACHE_HOME": str(cache_root),
        },
    )
    track_name = audio_path.stem
    base = out_dir / model / track_name
    vocals = base / "vocals.wav"
    background = base / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        raise RuntimeError("Demucs separation finished but expected stems were not found.")
    return vocals, background


def _slice_audio(input_wav: Path, output_wav: Path, start_ms: int, end_ms: int) -> None:
    audio, sr = sf.read(str(input_wav), always_2d=True)
    start = max(0, int(start_ms * sr / 1000))
    end = min(len(audio), int(end_ms * sr / 1000))
    if start >= end:
        raise ValueError("Invalid audio slice range.")
    sf.write(str(output_wav), audio[start:end], sr)


def _subdivide_range(start_ms: int, end_ms: int, max_segment_ms: int) -> list[tuple[int, int]]:
    duration = end_ms - start_ms
    if duration <= max_segment_ms:
        return [(start_ms, end_ms)]
    segments: list[tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        next_end = min(end_ms, cursor + max_segment_ms)
        segments.append((cursor, next_end))
        cursor = next_end
    return segments


def _reference_window(
    seg_start_ms: int,
    seg_end_ms: int,
    total_duration_ms: int,
    target_ms: int = DEFAULT_REFERENCE_TARGET_MS,
) -> tuple[int, int]:
    seg_duration = max(1, seg_end_ms - seg_start_ms)
    if seg_duration >= target_ms:
        return seg_start_ms, seg_end_ms

    extra = target_ms - seg_duration
    extend_before = extra // 2
    extend_after = extra - extend_before

    ref_start = max(0, seg_start_ms - extend_before)
    ref_end = min(total_duration_ms, seg_end_ms + extend_after)

    current = ref_end - ref_start
    if current < target_ms:
        shortage = target_ms - current
        if ref_start == 0:
            ref_end = min(total_duration_ms, ref_end + shortage)
        elif ref_end == total_duration_ms:
            ref_start = max(0, ref_start - shortage)

    return ref_start, ref_end


def _build_global_reference_clip(
    vocals_path: Path,
    items: list[TranscriptSegment] | list[DubChunk],
    output_wav: Path,
    target_ms: int = DEFAULT_GLOBAL_REFERENCE_TARGET_MS,
) -> tuple[int, int]:
    audio, sr = sf.read(str(vocals_path), always_2d=True)
    voiced_chunks: list[np.ndarray] = []
    used_ms = 0

    for item in items:
        start = max(0, int(item.start_ms * sr / 1000))
        end = min(len(audio), int(item.end_ms * sr / 1000))
        if end <= start:
            continue
        chunk = audio[start:end]
        voiced_chunks.append(chunk)
        used_ms += item.duration_ms
        if used_ms >= target_ms:
            break

    if not voiced_chunks:
        sf.write(str(output_wav), audio, sr)
        return 0, int(len(audio) * 1000 / sr)

    reference = np.concatenate(voiced_chunks, axis=0)
    sf.write(str(output_wav), reference, sr)
    return 0, int(len(reference) * 1000 / sr)


def _clean_transcript_text(text: str) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    cleaned = re.sub(r"\s+([,.;:?!])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _join_chunk_text(parts: list[str]) -> str:
    return _clean_transcript_text(" ".join(part.strip() for part in parts if part and part.strip()))


def _should_flush_segment(text: str, duration_ms: int, gap_ms: int, max_segment_ms: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if duration_ms >= max_segment_ms:
        return True
    if gap_ms >= 550:
        return True
    if re.search(r"[.?!…)]$", stripped):
        return True
    if len(stripped) >= 80 and re.search(r"[,;:]$", stripped):
        return True
    return False


def _phoneme_budget(text: str, lang: str) -> int:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return 0
    try:
        if lang.startswith("en"):
            from misaki import en

            g2p = en.G2P(trf=False, british=False, fallback=None)
            phonemes, _ = g2p(cleaned)
            return len(re.sub(r"\s+", "", phonemes))
        if lang.startswith("ja"):
            from misaki import ja  # type: ignore

            g2p = ja.G2P()  # type: ignore[attr-defined]
            phonemes, _ = g2p(cleaned)
            return len(re.sub(r"\s+", "", phonemes))
        if lang.startswith("zh") or lang.startswith("yue"):
            from misaki import zh  # type: ignore

            g2p = zh.G2P()  # type: ignore[attr-defined]
            phonemes, _ = g2p(cleaned)
            return len(re.sub(r"\s+", "", phonemes))
    except Exception:
        pass
    return max(len(cleaned), math.ceil(len(cleaned) * 1.2))


def _atempo_filter_chain(rate: float) -> str:
    stages: list[float] = []
    remaining = float(rate)
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def _time_stretch_speech_ffmpeg(wav: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="voxcpm-atempo-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        src = tmp_root / "src.wav"
        dst = tmp_root / "dst.wav"
        sf.write(str(src), wav.astype(np.float32), sample_rate)
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-filter:a",
                _atempo_filter_chain(rate),
                str(dst),
            ]
        )
        stretched, _ = sf.read(str(dst), dtype="float32")
        return np.asarray(stretched, dtype=np.float32)


def _transcribe_segments_with_whisper_cpp(
    vocals_path: Path,
    language: str,
    whisper_cli: Path,
    whisper_model: Path,
    work_dir: Path,
    use_gpu: bool,
    max_segment_ms: int = DEFAULT_MAX_SEGMENT_MS,
) -> list[TranscriptSegment]:
    if not whisper_cli.exists():
        raise RuntimeError(f"whisper.cpp binary not found: {whisper_cli}")
    if not whisper_model.exists():
        raise RuntimeError(f"whisper.cpp model not found: {whisper_model}")

    whisper_input = work_dir / "vocals_16k_mono.wav"
    whisper_prefix = work_dir / "whisper_words"
    whisper_json = work_dir / "whisper_words.json"
    _resample_for_whisper(vocals_path, whisper_input)

    cmd = [
        str(whisper_cli),
        "-m",
        str(whisper_model),
        "-f",
        str(whisper_input),
        "-ojf",
        "-of",
        str(whisper_prefix),
        "-ml",
        "1",
        "-sow",
        "-pp",
    ]
    if not use_gpu:
        cmd.append("-ng")
    if language and language.lower() != "auto":
        cmd.extend(["-l", language])
    _run(cmd)
    if not whisper_json.exists():
        raise RuntimeError(f"whisper.cpp did not produce JSON output: {whisper_json}")

    data = json.loads(whisper_json.read_text(encoding="utf-8"))
    words = data.get("transcription", [])
    transcript_segments: list[TranscriptSegment] = []
    current_parts: list[str] = []
    seg_start_ms: Optional[int] = None
    seg_end_ms: Optional[int] = None
    prev_end_ms: Optional[int] = None

    def flush() -> None:
        nonlocal current_parts, seg_start_ms, seg_end_ms
        text = _clean_transcript_text("".join(current_parts))
        if text and seg_start_ms is not None and seg_end_ms is not None and seg_end_ms > seg_start_ms:
            transcript_segments.append(
                TranscriptSegment(
                    index=len(transcript_segments),
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    source_text=text,
                    duration_ms=seg_end_ms - seg_start_ms,
                    phoneme_budget=_phoneme_budget(text, language or "en"),
                )
            )
        current_parts = []
        seg_start_ms = None
        seg_end_ms = None

    for item in words:
        piece = item.get("text", "")
        offsets = item.get("offsets", {})
        start_ms = int(offsets.get("from", 0))
        end_ms = int(offsets.get("to", start_ms))
        cleaned_piece = piece.strip()
        if not cleaned_piece or cleaned_piece.startswith("[_"):
            continue

        if seg_start_ms is None:
            seg_start_ms = start_ms
        gap_ms = max(0, start_ms - prev_end_ms) if prev_end_ms is not None else 0
        current_parts.append(piece)
        seg_end_ms = end_ms
        prev_end_ms = end_ms

        joined = _clean_transcript_text("".join(current_parts))
        duration_ms = (seg_end_ms - seg_start_ms) if seg_start_ms is not None and seg_end_ms is not None else 0
        if _should_flush_segment(joined, duration_ms, gap_ms, max_segment_ms):
            flush()

    flush()
    return transcript_segments


def _merge_segments_into_chunks(
    segments: list[TranscriptSegment],
    source_lang: str,
    max_gap_ms: int = DEFAULT_CHUNK_MERGE_GAP_MS,
    max_chunk_duration_ms: int = DEFAULT_MAX_CHUNK_DURATION_MS,
) -> list[DubChunk]:
    if not segments:
        return []

    chunks: list[DubChunk] = []
    current_segments: list[TranscriptSegment] = [segments[0]]

    def flush() -> None:
        nonlocal current_segments
        if not current_segments:
            return
        start_ms = current_segments[0].start_ms
        end_ms = current_segments[-1].end_ms
        source_text = _join_chunk_text([seg.source_text for seg in current_segments])
        chunks.append(
            DubChunk(
                index=len(chunks),
                start_ms=start_ms,
                end_ms=end_ms,
                source_text=source_text,
                duration_ms=end_ms - start_ms,
                phoneme_budget=_phoneme_budget(source_text, source_lang),
                segment_indexes=[seg.index for seg in current_segments],
            )
        )
        current_segments = []

    for seg in segments[1:]:
        prev = current_segments[-1]
        gap_ms = max(0, seg.start_ms - prev.end_ms)
        prospective_start_ms = current_segments[0].start_ms
        prospective_end_ms = seg.end_ms
        prospective_duration_ms = prospective_end_ms - prospective_start_ms
        if gap_ms <= max_gap_ms and prospective_duration_ms <= max_chunk_duration_ms:
            current_segments.append(seg)
            continue
        flush()
        current_segments = [seg]

    flush()
    return chunks


def _translation_prompt(
    chunks: list[DubChunk],
    source_lang: str,
    target_lang: str,
    style_hint: str,
) -> dict[str, Any]:
    full_text = "\n".join(f"[{chunk.index}] {chunk.source_text}" for chunk in chunks)
    segment_payload = [
        {
            "index": chunk.index,
            "source_text": chunk.source_text,
            "duration_ms": chunk.duration_ms,
            "duration_seconds": round(chunk.duration_ms / 1000, 3),
            "phoneme_budget": chunk.phoneme_budget,
            "segment_indexes": chunk.segment_indexes,
            "target_phoneme_range": {
                "min": max(1, math.floor(chunk.phoneme_budget * 0.8)),
                "max": max(1, math.floor(chunk.phoneme_budget * DEFAULT_MAX_ALLOWED_SPEEDUP)),
            },
            "timing_hint": (
                "This is one continuous dubbing chunk. Translate the whole chunk naturally, "
                "but keep the spoken Chinese conservative in length. If timing is tight, prefer shorter wording."
            ),
        }
        for chunk in chunks
    ]
    instructions = {
        "task": "Read the full transcript first, then translate chunk by chunk.",
        "source_language": source_lang,
        "target_language": target_lang,
        "style_hint": style_hint,
        "rules": [
            "Return strict JSON only.",
            "Preserve chunk indexes exactly.",
            "Keep each translated chunk natural and dub-friendly.",
            "Treat phoneme_budget as a source-side speaking-length proxy and keep the Chinese spoken form conservative.",
            "If a chunk is dense, shorten wording instead of preserving every detail literally.",
            "Meaning should stay basically faithful overall, but local phrasing can be simplified for timing.",
            "For very continuous speech, it is better to use shorter, smoother narration than to overrun and force unnatural speed.",
        ],
        "response_schema": {
            "segments": [{"index": 0, "translated_text": "translated sentence"}],
        },
        "full_transcript": full_text,
        "segments": segment_payload,
    }
    return instructions


def _save_translation_prompt(
    chunks: list[DubChunk],
    source_lang: str,
    target_lang: str,
    style_hint: str,
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(_translation_prompt(chunks, source_lang, target_lang, style_hint), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _predicted_ratio(phoneme_budget: int, translated_phoneme_budget: int) -> float:
    if phoneme_budget <= 0 or translated_phoneme_budget <= 0:
        return 1.0
    return float(translated_phoneme_budget) / float(phoneme_budget)


def _language_calibration(lang: str) -> float:
    lang_prefix = _normalized_lang_prefix(lang)
    store = _load_calibration_store()
    entry = store.get("languages", {}).get(lang_prefix, {})
    factor = entry.get("phoneme_factor")
    if isinstance(factor, (int, float)):
        return float(factor)
    return PHONEME_RATIO_CALIBRATION.get(lang_prefix, 1.0)


def _predicted_min_ratio(lang: str) -> float:
    lang_prefix = _normalized_lang_prefix(lang)
    store = _load_calibration_store()
    entry = store.get("languages", {}).get(lang_prefix, {})
    min_ratio = entry.get("min_ratio")
    if isinstance(min_ratio, (int, float)):
        return float(min_ratio)
    return DEFAULT_MIN_PREDICTED_RATIO


def _effective_predicted_ratio(source_phoneme_budget: int, translated_phoneme_budget: int, target_lang: str) -> float:
    calibrated = translated_phoneme_budget * _language_calibration(target_lang)
    return _predicted_ratio(source_phoneme_budget, int(round(calibrated)))


def _predicted_timing_issue(
    chunk: DubChunk,
    *,
    target_lang: str = "zh",
    min_ratio: Optional[float] = None,
    max_ratio: float = DEFAULT_MAX_ALLOWED_SPEEDUP,
    retry_round: int = 1,
) -> Optional[dict[str, Any]]:
    if min_ratio is None:
        min_ratio = _predicted_min_ratio(target_lang)
    translated_phonemes = _phoneme_budget(chunk.translated_text or "", "zh")
    chunk.translated_phoneme_budget = translated_phonemes
    predicted_ratio = _effective_predicted_ratio(chunk.phoneme_budget, translated_phonemes, target_lang)
    if min_ratio <= predicted_ratio <= max_ratio:
        return None

    char_count = len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", chunk.translated_text or ""))
    shorten_factor = 0.78 if retry_round <= 1 else 0.55
    expand_factor = 1.18 if retry_round <= 1 else 1.32
    if predicted_ratio > max_ratio:
        direction = "shorten"
        max_chars = max(0, math.floor(char_count * shorten_factor))
        min_chars = 0
        severity = "severe" if predicted_ratio >= 1.22 else "mild"
        instruction = (
            "This Chinese chunk is predicted to be too long before TTS. Rewrite it shorter first. "
            "Keep only the core meaning, avoid literal carry-over, and make the spoken form clearly leaner."
        )
    else:
        direction = "expand"
        min_chars = max(char_count + 1, math.ceil(max(1, char_count) * expand_factor))
        max_chars = max(min_chars, char_count + 6)
        severity = "severe" if predicted_ratio <= 0.55 else "mild"
        instruction = (
            "This Chinese chunk is predicted to be too short before TTS. Expand it a bit so the spoken form feels complete, "
            "but keep it natural and still within the target phoneme budget."
        )

    return {
        "index": chunk.index,
        "retry_round": retry_round,
        "direction": direction,
        "severity": severity,
        "source_text": chunk.source_text,
        "translated_text": chunk.translated_text,
        "duration_ms": chunk.duration_ms,
        "source_phoneme_budget": chunk.phoneme_budget,
        "translated_phoneme_budget": translated_phonemes,
        "predicted_ratio": round(predicted_ratio, 3),
        "target_phoneme_lower_hint": max(1, math.ceil(chunk.phoneme_budget * min_ratio)),
        "target_phoneme_upper_hint": max(1, math.floor(chunk.phoneme_budget * max_ratio)),
        "min_chars_hint": min_chars,
        "max_chars_hint": max_chars,
        "instruction": instruction,
    }


def _preflight_translation_issues(
    chunks: list[DubChunk],
    *,
    target_lang: str = "zh",
    min_ratio: Optional[float] = None,
    max_ratio: float = DEFAULT_MAX_ALLOWED_SPEEDUP,
    retry_round: int = 1,
) -> list[dict[str, Any]]:
    if min_ratio is None:
        min_ratio = _predicted_min_ratio(target_lang)
    issues: list[dict[str, Any]] = []
    for chunk in chunks:
        issue = _predicted_timing_issue(
            chunk,
            target_lang=target_lang,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            retry_round=retry_round,
        )
        if issue:
            issues.append(issue)
    return issues


def _duration_ratio(item: TranscriptSegment | DubChunk) -> float:
    raw_ms = item.raw_generated_duration_ms or item.rendered_duration_ms
    if item.duration_ms <= 0 or raw_ms <= 0:
        return 1.0
    return float(raw_ms) / float(item.duration_ms)


def _problem_segments(
    chunks: list[DubChunk],
    target_lang: str = "zh",
    mild_ratio: float = DEFAULT_MAX_ALLOWED_SPEEDUP,
    severe_ratio: float = 1.22,
    retry_round: int = 1,
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for chunk in chunks:
        preflight_issue = _predicted_timing_issue(
            chunk,
            target_lang=target_lang,
            min_ratio=_predicted_min_ratio(target_lang),
            max_ratio=mild_ratio,
            retry_round=retry_round,
        )
        actual_ratio = _duration_ratio(chunk)
        if actual_ratio <= mild_ratio and not preflight_issue:
            continue
        if preflight_issue and preflight_issue["direction"] == "expand":
            problems.append(preflight_issue | {"stage": "preflight"})
            continue

        translated_phonemes = chunk.translated_phoneme_budget or _phoneme_budget(chunk.translated_text or "", "zh")
        predicted_ratio = _effective_predicted_ratio(chunk.phoneme_budget, translated_phonemes, target_lang)
        ratio = max(actual_ratio, predicted_ratio)
        base_issue = preflight_issue or {}
        problems.append(
            {
                "index": chunk.index,
                "retry_round": retry_round,
                "direction": base_issue.get("direction", "shorten"),
                "severity": "severe" if ratio >= severe_ratio else "mild",
                "stage": "rendered",
                "source_text": chunk.source_text,
                "translated_text": chunk.translated_text,
                "duration_ms": chunk.duration_ms,
                "raw_generated_duration_ms": chunk.raw_generated_duration_ms,
                "duration_ratio": round(ratio, 3),
                "source_phoneme_budget": chunk.phoneme_budget,
                "translated_phoneme_budget": translated_phonemes,
                "predicted_ratio": round(predicted_ratio, 3),
                "min_chars_hint": base_issue.get("min_chars_hint", 0),
                "max_chars_hint": base_issue.get(
                    "max_chars_hint",
                    max(0, math.floor(len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", chunk.translated_text or "")) * (0.78 if retry_round <= 1 else 0.55))),
                ),
                "target_phoneme_lower_hint": base_issue.get(
                    "target_phoneme_lower_hint",
                    max(1, math.ceil(chunk.phoneme_budget * _predicted_min_ratio(target_lang))),
                ),
                "target_phoneme_upper_hint": base_issue.get(
                    "target_phoneme_upper_hint",
                    max(1, math.floor(chunk.phoneme_budget * mild_ratio)),
                ),
                "instruction": base_issue.get(
                    "instruction",
                    "This rendered chunk still overruns. Rewrite it shorter first, then render again only after the predicted phoneme ratio looks safe.",
                ),
            }
        )
    return problems


def _save_retry_translation_prompt(
    chunks: list[DubChunk],
    output_path: Path,
    target_lang: str = "zh",
    retry_round: int = 1,
) -> None:
    round_rule = (
        "First retry: shorten aggressively and aim below the hinted max_chars_hint."
        if retry_round <= 1
        else "Second retry: shorten even more aggressively; if needed reduce to one tiny phrase or an empty string."
    )
    payload = {
        "task": "Only rewrite the problematic Chinese dubbing chunks.",
        "rules": [
            "Return strict JSON only.",
            "Only include listed indexes.",
            "The whole narration only needs to stay smooth and basically faithful overall; individual words do not need literal preservation.",
            "First satisfy the predicted phoneme range before asking VoxCPM to render again.",
            "If direction is shorten, make the Chinese chunk clearly shorter and easier to say quickly.",
            "If direction is expand, add a little more natural spoken content so the chunk is not obviously too thin.",
            "Use the phoneme and character hints as a pre-check: if the rewrite still looks out of range, adjust it again before re-rendering.",
            round_rule,
        ],
        "segments": _problem_segments(chunks, target_lang=target_lang, retry_round=retry_round),
        "response_schema": {
            "segments": [{"index": 0, "translated_text": "shorter rewritten line"}],
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_preflight_translation_prompt(
    chunks: list[DubChunk],
    output_path: Path,
    target_lang: str = "zh",
    retry_round: int = 1,
) -> None:
    payload = {
        "task": "Rewrite only the Chinese chunks whose predicted phoneme length is out of range, before any TTS rendering.",
        "rules": [
            "Return strict JSON only.",
            "Only include listed indexes.",
            "Do not ask for TTS yet; this is a prediction-only timing pass.",
            "If direction is shorten, simplify hard and keep only the essential meaning.",
            "If direction is expand, add a little natural detail so the spoken line does not feel too empty.",
            "The full narration only needs to stay smooth and basically faithful overall.",
            "Use the phoneme hints as the primary constraint before cloning.",
        ],
        "segments": _preflight_translation_issues(chunks, target_lang=target_lang, retry_round=retry_round),
        "response_schema": {
            "segments": [{"index": 0, "translated_text": "rewritten chunk"}],
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fit_audio_to_duration(
    wav: np.ndarray,
    sample_rate: int,
    target_ms: int,
    max_allowed_speedup: float = DEFAULT_MAX_ALLOWED_SPEEDUP,
) -> np.ndarray:
    target_samples = max(1, int(sample_rate * target_ms / 1000))
    if len(wav) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    wav = wav.astype(np.float32)
    current = len(wav)

    if current > target_samples:
        ratio = current / max(1, target_samples)
        if ratio <= max_allowed_speedup:
            try:
                stretched = _time_stretch_speech_ffmpeg(wav, sample_rate, ratio)
                if len(stretched) >= target_samples:
                    return stretched[:target_samples]
                pad = np.zeros(target_samples - len(stretched), dtype=np.float32)
                return np.concatenate([stretched, pad])
            except Exception:
                pass

        try:
            stretched = _time_stretch_speech_ffmpeg(wav, sample_rate, max_allowed_speedup)
            if len(stretched) >= target_samples:
                return stretched[:target_samples]
            pad = np.zeros(target_samples - len(stretched), dtype=np.float32)
            return np.concatenate([stretched, pad])
        except Exception:
            pass

        trimmed = wav[:target_samples].copy()
        fade = min(int(sample_rate * 0.04), max(1, target_samples // 8))
        if fade > 1:
            trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return trimmed

    if current < target_samples:
        pad = np.zeros(target_samples - current, dtype=np.float32)
        return np.concatenate([wav, pad])

    return wav


def _apply_edge_fade(wav: np.ndarray, sample_rate: int, fade_ms: int = 12) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).copy()
    if len(wav) <= 2:
        return wav
    fade = min(int(sample_rate * fade_ms / 1000), len(wav) // 2)
    if fade <= 1:
        return wav
    ramp_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    wav[:fade] *= ramp_in
    wav[-fade:] *= ramp_out
    return wav


def _render_dub_track(
    chunks: list[DubChunk],
    vocals_path: Path,
    output_wav: Path,
    model_id: str,
    control_hint: str,
    device: str = "auto",
    global_reference_target_ms: int = DEFAULT_GLOBAL_REFERENCE_TARGET_MS,
) -> list[DubChunk]:
    total_duration = _ffprobe_duration(vocals_path)
    engine = VoxCPMEngine(EngineConfig(model_id=model_id, device=device, optimize=False, enable_denoiser=False))
    sample_rate = engine.backend.get_model().tts_model.sample_rate
    canvas = np.zeros(int(total_duration * sample_rate) + sample_rate, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="voxcpm-dub-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        global_ref_clip = tmp_root / "global_reference.wav"
        _build_global_reference_clip(vocals_path, chunks, global_ref_clip, global_reference_target_ms)
        for idx, chunk in enumerate(chunks):
            text = (chunk.translated_text or "").strip()
            if not text:
                continue
            chunk.translated_phoneme_budget = _phoneme_budget(text, DEFAULT_TARGET_LANGUAGE)
            request = GenerationRequest(
                text=text,
                control=control_hint,
                reference_audio=str(global_ref_clip),
                inference_timesteps=10,
                cfg_value=2.0,
                normalize=True,
            )
            sr, wav = engine.generate(request)
            if sr != sample_rate:
                raise RuntimeError(f"Unexpected sample rate change during dubbing: {sample_rate} -> {sr}")
            raw_wav = np.asarray(wav, dtype=np.float32)
            chunk.raw_generated_duration_ms = int(len(raw_wav) * 1000 / sample_rate)
            wav = _fit_audio_to_duration(raw_wav, sample_rate, chunk.duration_ms)
            wav = _apply_edge_fade(wav, sample_rate)
            start_sample = int(chunk.start_ms * sample_rate / 1000)
            next_start_ms = chunks[idx + 1].start_ms if idx + 1 < len(chunks) else int(total_duration * 1000)
            max_end_sample = min(len(canvas), int(next_start_ms * sample_rate / 1000))
            end_sample = min(max_end_sample, start_sample + len(wav))
            mixed_len = max(0, end_sample - start_sample)
            if mixed_len == 0:
                continue
            canvas[start_sample:end_sample] += wav[:mixed_len]
            chunk.rendered_duration_ms = int(len(wav) * 1000 / sample_rate)
            chunk.adjusted_duration_ms = int(mixed_len * 1000 / sample_rate)
            chunk.timeline_start_ms = int(start_sample * 1000 / sample_rate)
            chunk.timeline_end_ms = int(end_sample * 1000 / sample_rate)

    canvas = np.clip(canvas, -0.98, 0.98)
    sf.write(str(output_wav), canvas, sample_rate)
    return chunks


def _mix_and_mux(video_path: Path, background_path: Path, dub_wav: Path, output_video: Path) -> None:
    _require_binary("ffmpeg")
    mixed_audio = output_video.with_suffix(".mixed.wav")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(background_path),
            "-i",
            str(dub_wav),
            "-filter_complex",
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[bg];"
            "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[dub];"
            "[bg][dub]amix=inputs=2:normalize=0",
            str(mixed_audio),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(mixed_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-shortest",
            str(output_video),
        ]
    )


def run_pipeline(args: argparse.Namespace) -> Path:
    work_dir = Path(args.output_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video).resolve()
    audio_path = work_dir / "source_audio.wav"
    separated_dir = work_dir / "separated"
    dub_track = work_dir / "dub_track.wav"
    manifest_path = work_dir / "manifest.json"
    output_video = work_dir / f"{video_path.stem}.dubbed.mp4"

    _extract_audio(video_path, audio_path)
    vocals_path, background_path = _separate_audio(audio_path, separated_dir, args.demucs_model, args.demucs_device)
    segments = _transcribe_segments_with_whisper_cpp(
        vocals_path,
        args.source_language,
        Path(args.whisper_cli).resolve(),
        Path(args.whisper_model).resolve(),
        work_dir,
        args.whisper_gpu,
        args.max_segment_ms,
    )
    if not segments:
        raise RuntimeError("No speech segments detected.")
    chunks = _merge_segments_into_chunks(
        segments,
        args.source_language,
        args.chunk_merge_gap_ms,
        args.max_chunk_duration_ms,
    )
    prompt_path = work_dir / "translation_request.json"
    _save_translation_prompt(chunks, args.source_language, args.target_language, args.style_hint, prompt_path)
    raise RuntimeError(
        "Translation request prepared for agent-driven translation. "
        f"Read {prompt_path} together with skills/short-video-dubbing/references/translation_rules.md, "
        "fill translated_text for each chunk, then re-run rendering."
    )

    chunks = _render_dub_track(
        chunks,
        vocals_path=vocals_path,
        output_wav=dub_track,
        model_id=args.model_id,
        control_hint=args.tts_control,
    )
    _mix_and_mux(video_path, background_path, dub_track, output_video)

    manifest = {
        "video": str(video_path),
        "vocals": str(vocals_path),
        "background": str(background_path),
        "dub_track": str(dub_track),
        "output_video": str(output_video),
        "asr_backend": "whisper.cpp",
        "whisper_cli": str(Path(args.whisper_cli).resolve()),
        "whisper_model": str(Path(args.whisper_model).resolve()),
        "whisper_gpu": bool(args.whisper_gpu),
        "cloning_mode": "chunkwise_controllable_clone",
        "segments": [asdict(seg) for seg in segments],
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short video dubbing translation pipeline for VoxCPM.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="./outputs/video_dub", help="Working directory for artifacts")
    parser.add_argument("--source-language", default="en", help="whisper.cpp source language hint")
    parser.add_argument("--target-language", required=True, help="Target dubbing language")
    parser.add_argument("--style-hint", default="Natural short-video dubbing, conversational but time-aligned.")
    parser.add_argument("--demucs-model", default=DEFAULT_DEMUCS_MODEL)
    parser.add_argument("--demucs-device", default="cpu")
    parser.add_argument("--model-id", default=str(Path.cwd() / "models" / "VoxCPM2"))
    parser.add_argument("--tts-control", default="", help="Extra voice design prompt for VoxCPM")
    parser.add_argument(
        "--whisper-cli",
        default=str(Path.cwd() / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        help="Path to the whisper.cpp CLI binary",
    )
    parser.add_argument(
        "--whisper-model",
        default=str(Path.cwd() / "tools" / "whisper.cpp" / "models" / "ggml-medium.en.bin"),
        help="Path to the whisper.cpp model file",
    )
    parser.add_argument("--whisper-gpu", action="store_true", help="Use whisper.cpp GPU/Metal path when stable")
    parser.add_argument("--max-segment-ms", type=int, default=DEFAULT_MAX_SEGMENT_MS)
    parser.add_argument("--chunk-merge-gap-ms", type=int, default=DEFAULT_CHUNK_MERGE_GAP_MS)
    parser.add_argument("--max-chunk-duration-ms", type=int, default=DEFAULT_MAX_CHUNK_DURATION_MS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = run_pipeline(args)
    print(output)


if __name__ == "__main__":
    main()
