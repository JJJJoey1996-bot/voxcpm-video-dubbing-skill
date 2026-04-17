#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from voxcpm.video_dub import (
    TranscriptSegment,
    _clean_transcript_text,
    _merge_segments_into_chunks,
    _mix_and_mux,
    _phoneme_budget,
    _render_dub_track,
    _should_flush_segment,
    update_calibration_from_manifests,
)


def _segments_from_whisper_json(path: Path, source_language: str, max_segment_ms: int) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = data.get("transcription", [])
    segments: list[TranscriptSegment] = []
    current_parts: list[str] = []
    seg_start_ms: int | None = None
    seg_end_ms: int | None = None
    prev_end_ms: int | None = None

    def flush() -> None:
        nonlocal current_parts, seg_start_ms, seg_end_ms
        text = _clean_transcript_text("".join(current_parts))
        if text and seg_start_ms is not None and seg_end_ms is not None and seg_end_ms > seg_start_ms:
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    source_text=text,
                    duration_ms=seg_end_ms - seg_start_ms,
                    phoneme_budget=_phoneme_budget(text, source_language),
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
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a prepared short-video dubbing job from translated chunk JSON.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--probe-dir", required=True, help="Directory containing whisper_words.json and separated stems")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--translated-json", required=True, help="JSON with translated chunks keyed by index")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--chunk-merge-gap-ms", type=int, default=350)
    parser.add_argument("--max-chunk-duration-ms", type=int, default=18000)
    parser.add_argument("--max-segment-ms", type=int, default=4500)
    parser.add_argument("--model-id", default="./models/VoxCPM2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tts-control", default="Natural short-video dubbing in Chinese, stable single-speaker voice.")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    probe_dir = Path(args.probe_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = _segments_from_whisper_json(probe_dir / "whisper_words.json", args.source_language, args.max_segment_ms)
    chunks = _merge_segments_into_chunks(
        segments,
        args.source_language,
        args.chunk_merge_gap_ms,
        args.max_chunk_duration_ms,
    )

    translated_payload = json.loads(Path(args.translated_json).read_text(encoding="utf-8"))
    translated_map = {int(item["index"]): item["translated_text"] for item in translated_payload["chunks"]}
    for chunk in chunks:
        chunk.translated_text = translated_map.get(chunk.index, "").strip()

    vocals_path = probe_dir / "separated" / "mdx_q" / "source_audio" / "vocals.wav"
    background_path = probe_dir / "separated" / "mdx_q" / "source_audio" / "no_vocals.wav"
    dub_track = output_dir / "dub_track.wav"
    output_video = output_dir / f"{video_path.stem}.zh.safe.mp4"

    rendered_chunks = _render_dub_track(
        chunks,
        vocals_path=vocals_path,
        output_wav=dub_track,
        model_id=args.model_id,
        control_hint=args.tts_control,
        device=args.device,
    )
    _mix_and_mux(video_path, background_path, dub_track, output_video)

    manifest = {
        "video": str(video_path),
        "probe_dir": str(probe_dir),
        "output_video": str(output_video),
        "dub_track": str(dub_track),
        "translated_json": str(Path(args.translated_json).resolve()),
        "safety_mode": {
            "chunk_merge_gap_ms": args.chunk_merge_gap_ms,
            "max_chunk_duration_ms": args.max_chunk_duration_ms,
            "sequential_render": True,
        },
        "chunks": [asdict(chunk) for chunk in rendered_chunks],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration = update_calibration_from_manifests("zh")
    print(json.dumps({"output_video": str(output_video), "calibration": calibration}, ensure_ascii=False))
    print(output_video)


if __name__ == "__main__":
    main()
