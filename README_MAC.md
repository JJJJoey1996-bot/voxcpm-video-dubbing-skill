# VoxCPM2 Mac Studio

This fork is now focused on a single stable path: `PyTorch + MPS` on Apple Silicon.

## What changed

- Removes the experimental MLX runtime path and keeps one inference backend.
- Uses local model directories by default for faster repeated startup.
- Adds a reusable download script with two sources:
  - Hugging Face: `openbmb/VoxCPM2`
  - ModelScope: `OpenBMB/VoxCPM2`
- Keeps the richer WebUI for voice design, controllable cloning, and ultimate cloning.

## Quick start

```bash
./scripts/setup_mac.sh
./scripts/run_webui_mac.sh
```

Open `http://127.0.0.1:8808` after startup.

## Download sources

- Hugging Face: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- ModelScope: [OpenBMB/VoxCPM2](https://modelscope.cn/models/OpenBMB/VoxCPM2)

## One-command download

```bash
./scripts/download_model.sh
```

Optional environment variables:

```bash
VOXCPM_DOWNLOAD_SOURCE=hf
VOXCPM_DOWNLOAD_SOURCE=modelscope
VOXCPM_MODEL_DIR=/absolute/path/to/VoxCPM2
```

## Startup notes

- Default startup uses `--device mps`.
- Default startup points `--model-id` at `./models/VoxCPM2` once downloaded.
- If the local model directory is missing, the launcher automatically downloads it first.
- The one-click starter no longer runs `uv sync` on every launch. Set `VOXCPM_SYNC_ON_START=1` only when you really want to resync dependencies.

## MPS self-check and repair

Check whether PyTorch can really use the Apple GPU:

```bash
.venv/bin/python scripts/check_mps.py
```

If `mps_available` is `false`, try a targeted reinstall:

```bash
./scripts/repair_mps.sh
```

The repair script reinstalls `torch` and `torchaudio`, then reruns the MPS self-check.
