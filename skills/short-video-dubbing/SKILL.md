---
name: short-video-dubbing
description: Build or run a cross-platform short-video dubbing translation pipeline in this VoxCPM project. Use when the task involves installing VoxCPM video-dub dependencies, downloading whisper.cpp and VoxCPM models, extracting video audio, separating vocals/background, transcribing and timestamping speech with whisper.cpp, generating phoneme-budgeted translation prompts with Misaki, cloning speech with VoxCPM, and muxing the dubbed track back into video.
---

# Short Video Dubbing

Use this skill when the user wants end-to-end short-video dubbing translation based on VoxCPM.
It is written so Codex, Claude Code, or similar shell-capable agents can follow the same workflow.
It assumes the surrounding workspace contains this VoxCPM repository, not just the skill folder by itself.

## Workflow

1. Install everything first.
   - macOS / Linux:
     `./skills/short-video-dubbing/scripts/setup.sh`
   - Windows PowerShell:
     `powershell -ExecutionPolicy Bypass -File .\skills\short-video-dubbing\scripts\setup.ps1`
   - Cross-platform Python fallback:
     `python ./skills/short-video-dubbing/scripts/install.py`
2. For agent-friendly orchestration, prefer the unified wrapper:
   `python ./skills/short-video-dubbing/scripts/agent_pipeline.py install`
3. Prepare the job:
   `python ./skills/short-video-dubbing/scripts/agent_pipeline.py prepare --video INPUT.mp4 --output-dir ./outputs/job_name`
4. Read `translation_request.json` and `references/translation_rules.md`, then write a `translated_chunks.json` file.
5. Render the final dubbed video:
   `python ./skills/short-video-dubbing/scripts/agent_pipeline.py render --video INPUT.mp4 --probe-dir ./outputs/job_name --output-dir ./outputs/job_name_safe --translated-json ./outputs/job_name_safe/translated_chunks.json --device auto`

## Notes

- `scripts/install.py` is the cross-platform bootstrapper. It will:
  - install `uv` if missing
  - automatically provision a managed Python 3.12 `.venv` when the host Python is too new or otherwise unsuitable
  - install Python dependencies with `uv sync --extra video_dub`
  - install or prompt for `git`, `cmake`, and `ffmpeg`
  - clone and build `whisper.cpp`
  - download `ggml-medium.en.bin`
  - download `openbmb/VoxCPM2` into `./models/VoxCPM2`
- The short-video dubbing flow does not depend on `funasr`. `funasr` is now an optional dependency only for reference-audio auto-transcription in the interactive app, and that optional path is intended for Python versions below 3.13.
- This fork is intentionally trimmed for agent usage. The bundled Web UI path is not part of the supported install flow here.
- The pipeline uses `demucs` for two-stem separation (`vocals` and `no_vocals`).
- Speech timing now comes from `whisper.cpp`. The default path keeps GPU off for stability with `medium.en` on this Mac, but still gives cleaner native timestamp output than the old Python ASR chain here. The pipeline groups word timestamps into short sentence-like chunks to reduce drift.
- Misaki is used to estimate a phoneme budget for each dubbing chunk and include that constraint in the translation request.
- Translation is agent-driven, not API-driven. The agent should read `translation_request.json` together with `references/translation_rules.md`.
- VoxCPM synthesis should use chunk-by-chunk controllable cloning for the current single-speaker workflow, not many tiny line-by-line clones.
- Adjacent transcript pieces with very short gaps should be treated as one dubbing chunk, but long continuous speech still needs a safe chunk-duration ceiling on Mac instead of becoming one giant render.
- Before any clone render, translated chunks should first pass a phoneme-length prediction check; obvious overlong or undershort text should be rewritten before spending time on TTS.
- Retry and shortening decisions should happen at the chunk level first, before spending more time on another expensive clone render.
- On Apple Silicon, prefer `--device mps` during final render once MPS self-check passes.
- On Windows and Linux, `--device auto` is usually the right default; it will fall back to CUDA or CPU depending on the machine.

## References

- Implementation details: `references/pipeline.md`
- Translation rules: `references/translation_rules.md`
- Cross-platform installer: `scripts/install.py`
- Agent wrapper: `scripts/agent_pipeline.py`
