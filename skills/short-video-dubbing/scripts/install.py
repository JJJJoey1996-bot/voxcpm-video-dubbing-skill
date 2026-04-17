#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WHISPER_REPO = "https://github.com/ggml-org/whisper.cpp.git"
WHISPER_MODEL_REPO = "ggerganov/whisper.cpp"
WHISPER_MODEL_FILE = "ggml-medium.en.bin"
VOXCPM_HF_REPO = "openbmb/VoxCPM2"
VOXCPM_MODELSCOPE_REPO = "OpenBMB/VoxCPM2"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def which(name: str) -> str | None:
    return shutil.which(name)


def venv_python() -> Path:
    if platform.system() == "Windows":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def ensure_uv() -> None:
    if which("uv"):
        return
    run([sys.executable, "-m", "pip", "install", "uv"])


def install_system_packages() -> None:
    missing = [name for name in ("git", "cmake", "ffmpeg") if not which(name)]
    if not missing:
        return

    system = platform.system()
    if system == "Darwin" and which("brew"):
        run(["brew", "install", *missing])
        return
    if system == "Linux" and which("apt-get"):
        run(["sudo", "apt-get", "update"])
        run(["sudo", "apt-get", "install", "-y", *missing])
        return
    if system == "Windows" and which("winget"):
        package_map = {
            "git": "Git.Git",
            "cmake": "Kitware.CMake",
            "ffmpeg": "Gyan.FFmpeg",
        }
        for item in missing:
            run(
                [
                    "winget",
                    "install",
                    "--id",
                    package_map[item],
                    "-e",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ]
            )
        return
    raise RuntimeError(
        "Missing system tools: "
        + ", ".join(missing)
        + ". Install them manually or rerun on a machine with brew/apt-get/winget available."
    )


def ensure_python_stack() -> None:
    cache_dir = os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")
    run(["uv", "sync", "--extra", "video_dub"], cwd=REPO_ROOT)
    os.environ["UV_CACHE_DIR"] = cache_dir


def ensure_whisper_cpp() -> None:
    whisper_dir = REPO_ROOT / "tools" / "whisper.cpp"
    if not whisper_dir.exists():
        whisper_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", WHISPER_REPO, str(whisper_dir)])

    build_dir = whisper_dir / "build"
    cmake_args = ["cmake", "-S", str(whisper_dir), "-B", str(build_dir)]
    if platform.system() == "Darwin":
        cmake_args.append("-DGGML_METAL=ON")
    run(cmake_args)
    run(["cmake", "--build", str(build_dir), "--config", "Release"])

    model_dir = whisper_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / WHISPER_MODEL_FILE
    if model_path.exists():
        return

    py = venv_python()
    code = (
        "from huggingface_hub import hf_hub_download;"
        f"hf_hub_download(repo_id='{WHISPER_MODEL_REPO}', filename='{WHISPER_MODEL_FILE}', "
        f"local_dir=r'{model_dir}', local_dir_use_symlinks=False)"
    )
    run([str(py), "-c", code], cwd=REPO_ROOT)


def ensure_voxcpm_model(download_source: str) -> None:
    target_dir = REPO_ROOT / "models" / "VoxCPM2"
    if (target_dir / "config.json").exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    py = venv_python()
    if download_source in {"modelscope", "ms"}:
        code = (
            "from modelscope.hub.snapshot_download import snapshot_download;"
            f"snapshot_download(model_id='{VOXCPM_MODELSCOPE_REPO}', local_dir=r'{target_dir}')"
        )
    else:
        code = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download(repo_id='{VOXCPM_HF_REPO}', local_dir=r'{target_dir}', local_dir_use_symlinks=False)"
        )
    run([str(py), "-c", code], cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform installer for the short-video dubbing skill.")
    parser.add_argument("--download-source", default=os.environ.get("VOXCPM_DOWNLOAD_SOURCE", "hf"))
    parser.add_argument("--skip-system-deps", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")

    if not args.skip_system_deps:
        install_system_packages()
    ensure_uv()
    ensure_python_stack()
    ensure_whisper_cpp()
    ensure_voxcpm_model(args.download_source)

    print("short-video-dubbing skill is ready.")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Python: {venv_python()}")


if __name__ == "__main__":
    main()
